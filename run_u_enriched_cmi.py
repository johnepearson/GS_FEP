"""
Substrate-enriched blanket test (reviewer request): does adding near-boundary substrate (u) information
to the conditioning set make a flux-defined blanket screen the interior from the exterior, as measured by
the NONLINEAR conditional mutual information with a calibrated null?

Specification fixed in advance (no search over conditioning variables):
    Z_n = (flux_u, flux_v) + the first n mean-u shells beyond the organism boundary,
    shells = 0-6, 6-12, 12-18, 18-24, 24-36, 36-48, 48-64 px (columns 0..6 of u_shells),
    n in {0, 1, 2, 3, 5, 7}.
Statistic: I(I ; E | Z_n), Frenzel-Pompe k-NN estimator (k=5), every column z-scored before estimation.
Two nulls, both built to preserve p(E | Z_n) approximately:
    "cluster": shuffle E within 30 k-means clusters of Z_n (the paper's coarse null)
    "local":   Runge-style, swap E only among each point's 10 nearest neighbours in Z_n
Controls:
    "noise":   flux + n columns of pure noise (each u-shell column independently permuted), n=3 and 7. This shows
               what the estimator does when the added dimensions carry no information. If the real u-shells
               reduce the excess by more than noise columns do, the reduction is information, not dimension.
Subsets: all organisms, and isolated ones (merge_size == 1).
Same subsample (N_SUB rows) for every level, so levels are directly comparable.

Usage (data: master_trajectory_progress.npz from run_master_trajectory.py):
    python run_u_enriched_cmi.py --quick          # tiny test, a few minutes
    python run_u_enriched_cmi.py --jobs 6         # full run; --n-null 100 default, --n-sub 8000 default
Writes u_enriched_cmi.npz and prints a table. Resumable: finished (subset, condition) results are cached
in u_enriched_cmi_cache.npz.
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from conditional_mi import cmi_ksg

CACHE = "u_enriched_cmi_cache.pkl"  # keys include the target, so ring and far runs share it safely
LEVELS = [0, 1, 2, 3, 5, 7]
NOISE_LEVELS = [3, 7]


def zscore(a):
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a[:, None]
    return (a - a.mean(0)) / a.std(0)


def build_Z(flux, ush, kind, n, rng_seed, Ez=None, r2=None):
    if kind == "u":
        cols = [flux] + ([ush[:, :n]] if n > 0 else [])
    elif kind == "ecopy":
        # control: flux + a noisy copy of E whose squared correlation with E equals r2 (the R^2 the n real u-shells
        # achieve), padded with n-1 pure-noise columns so the dimension matches the real-u condition
        r = np.random.default_rng(rng_seed + 77)
        copy = Ez + r.normal(size=len(Ez)) * np.sqrt((1 - r2) / r2)
        pad = np.column_stack([r.permutation(ush[:, j]) for j in range(n - 1)])
        cols = [flux, copy[:, None], pad]
    else:  # noise columns: each u-shell column independently permuted
        r = np.random.default_rng(rng_seed)
        noise = np.column_stack([r.permutation(ush[:, j]) for j in range(n)])
        cols = [flux, noise]
    return zscore(np.column_stack(cols))


def cluster_null(Ii, Ei, Z, n_null, seed):
    km = KMeans(n_clusters=30, n_init=5, random_state=seed).fit(Z)
    lab = km.labels_
    groups = [np.where(lab == c)[0] for c in range(30)]
    out = np.empty(n_null)
    for t in range(n_null):
        r = np.random.default_rng(1000 + t)
        Es = Ei.copy()
        for g in groups:
            if len(g) > 1:
                Es[g] = Ei[g[r.permutation(len(g))]]
        out[t] = cmi_ksg(Ii, Es, Z)
    return out


def local_null(Ii, Ei, Z, n_null, seed, k_nb=10):
    tree = cKDTree(Z)
    _, nb = tree.query(Z, k=k_nb + 1, p=np.inf)
    nb = nb[:, 1:]
    out = np.empty(n_null)
    N = len(Ii)
    for t in range(n_null):
        r = np.random.default_rng(5000 + t)
        used = np.zeros(N, bool)
        src = np.arange(N)
        for i in r.permutation(N):
            cand = nb[i][~used[nb[i]]]
            if len(cand):
                j = cand[r.integers(len(cand))]
                src[i] = j
                used[j] = True
        out[t] = cmi_ksg(Ii, Ei[src], Z)
    return out


def task(args):
    name, Ii, Ei, flux, ush, kind, n, n_null, seed, r2 = args
    # every variable is z-scored (I, E, and each column of Z), as in the paper's primary CMI analysis
    Ii, Ei = zscore(Ii)[:, 0], zscore(Ei)[:, 0]
    Z = build_Z(flux, ush, kind, n, seed, Ez=Ei, r2=r2)
    t0 = time.time()
    obs = cmi_ksg(Ii, Ei, Z)
    nc = cluster_null(Ii, Ei, Z, n_null, seed)
    nl = local_null(Ii, Ei, Z, n_null, seed)
    return (name, kind, n), dict(obs=obs, null_cluster=nc, null_local=nl, secs=time.time() - t0, dim=Z.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--target", choices=["ring", "far"], default="ring",
                    help="ring: E = exterior ring mean v (0-24 px), as in the paper. far: E = mean v over shells 184-400 px "
                         "(9.2-18.2 radii), which does not overlap the u-shells used for conditioning")
    a = ap.parse_args()
    if a.quick:
        a.n_sub, a.n_null = 1500, 10

    d = np.load(a.data)
    I, E, fu, fv = d["I"], d["E"], d["fu"], d["fv"]
    if a.target == "far":
        E = np.nanmean(d["v_shells"][:, 21:30], axis=1)
    ush = d["u_shells"][:, :7]
    ms = d["merge_size"]
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(ush).all(1)
    print(f"{good.sum():,} organism-instances with all variables finite; isolated (merge_size==1): {(good & (ms == 1)).sum():,}")

    rng = np.random.default_rng(a.seed)
    subsets = {}
    for name, mask in (("all", good), ("isolated", good & (ms == 1))):
        idx = np.where(mask)[0]
        subsets[name] = rng.choice(idx, min(a.n_sub, len(idx)), replace=False)

    # diagnostic: how much of E and of I is linearly predictable from flux + u-shells (an E-proxy check)
    for name, idx in subsets.items():
        for lab, y in (("E", E[idx]), ("I", I[idx])):
            out_r2 = []
            for n in (0, 3, 7):
                X = np.column_stack([np.ones(len(idx)), fu[idx], fv[idx]] + ([ush[idx, :n]] if n else []))
                res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
                out_r2.append(1 - res.var() / y.var())
            print(f"R^2 of {lab} on flux + n u-shells [{name}], n=0/3/7: " + " / ".join(f"{x:.3f}" for x in out_r2))

    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    jobs = []
    for name, idx in subsets.items():
        flux = np.column_stack([fu[idx], fv[idx]])
        todo = [("u", n) for n in LEVELS] + [("noise", n) for n in NOISE_LEVELS] + [("ecopy", n) for n in (3, 7)]
        for kind, n in todo:
            if (a.target, name, kind, n, a.n_sub, a.n_null) in cache:
                continue
            r2 = None
            if kind == "ecopy":
                X = np.column_stack([np.ones(len(idx)), flux, ush[idx, :n]])
                Ez = (E[idx] - E[idx].mean()) / E[idx].std()
                r2 = 1 - (Ez - X @ np.linalg.lstsq(X, Ez, rcond=None)[0]).var()
            jobs.append((name, I[idx], E[idx], flux, ush[idx], kind, n, a.n_null, a.seed, r2))
    print(f"{len(jobs)} conditions to run ({a.n_null} null trials each, two nulls), {a.jobs} workers")
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for key, res in ex.map(task, jobs):
            cache[(a.target,) + key + (a.n_sub, a.n_null)] = res
            pickle.dump(cache, open(CACHE, "wb"))
            print(f"  done {key}  obs={res['obs']:.4f}  ({res['secs']:.0f}s; {time.time()-t0:.0f}s total)", flush=True)

    print("\nCMI I(I;E|Z), nats. excess = observed - null mean; SD units use the null SD.")
    print(f"{'subset':9s} {'cond':10s} {'dim':>3s} {'obs':>7s} | {'cluster null: mean':>18s} {'excess':>7s} {'nSD':>5s} | {'local null: mean':>16s} {'excess':>7s} {'nSD':>5s}")
    out = {}
    for name in subsets:
        for kind, n in [("u", n) for n in LEVELS] + [("noise", n) for n in NOISE_LEVELS] + [("ecopy", n) for n in (3, 7)]:
            r = cache[(a.target, name, kind, n, a.n_sub, a.n_null)]
            line = f"{name:9s} {kind+str(n):10s} {r['dim']:3d} {r['obs']:7.4f} |"
            for nk in ("null_cluster", "null_local"):
                m, s = r[nk].mean(), r[nk].std()
                line += f" {m:18.4f} {r['obs']-m:7.4f} {(r['obs']-m)/s:5.1f} |"
            print(line)
            out[f"{name}_{kind}{n}_obs"] = r["obs"]
            out[f"{name}_{kind}{n}_null_cluster"] = r["null_cluster"]
            out[f"{name}_{kind}{n}_null_local"] = r["null_local"]
    np.savez(f"u_enriched_cmi_{a.target}.npz", **out)
    print(f"\nwrote u_enriched_cmi_{a.target}.npz")


if __name__ == "__main__":
    main()
