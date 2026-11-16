"""
Blanket-thickness sweep (main text §3.5): is there a band of finite width w beyond the organism boundary
that screens the interior from the exterior just outside it?

Specification fixed in advance (no search over conditioning variables):
    Band of width w px beyond the contour, w in {0, 8, 16, 24, 40, 64, 96}.
    Blanket Z_w = (flux_u, flux_v) + 3 area-weighted mean-v summaries of the band + 3 mean-u summaries of the band
                  (the band is cut into thirds, on the available shell edges; u shells exist only to 64 px,
                   and only u shells lying wholly inside the band are used, so u summaries are absent for w < 12).
    Exterior E_w = area-weighted mean v over the 24-px shell immediately beyond the band, [w, w+24] px.
    w = 0 is the flux-only blanket with E = mean v over 0-24 px (shell-based, so it differs slightly from the
    paper's distance-transform ring E).
Statistic: Frenzel-Pompe CMI I(I ; E_w | Z_w), k = 5, every column z-scored (I, E_w and each column of Z_w).
Nulls: the paper's 30-cluster null and a Runge-style local-neighbour null, 100 trials each.
Controls at every w (same dimension as the real Z_w):
    noise: the band columns independently permuted (same marginals, no information)
    ecopy: flux + a noisy copy of E_w whose squared correlation with E_w equals the R^2 of the real band
           columns for E_w, padded with permuted columns to the same dimension
Subsets: all organisms, and isolated ones (merge_size == 1). Same N=8000 subsample for every w.

Usage (data: master_trajectory_progress.npz):
    python run_band_sweep_cmi.py --quick
    python run_band_sweep_cmi.py --jobs 12
Resumable (cache run_band_sweep_cache.pkl). Writes run_band_sweep.npz and prints a table.
"""
import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from conditional_mi import cmi_ksg
from run_u_enriched_cmi import zscore, cluster_null, local_null

CACHE = "run_band_sweep_cache.pkl"
WIDTHS = [0, 8, 16, 24, 40, 64, 96]
RADII = list(range(0, 161, 8)) + list(range(160, 401, 24))[1:]   # v-shell edges (px from the mask)
U_EDGES = [0, 6, 12, 18, 24, 36, 48, 64]
R_ORG = 21.3  # typical organism radius, for area weights ~ (R + mid-distance)


def wmean(vals, edges, lo, hi):
    """Area-weighted mean over the bins of `vals` (n x bins) with edges `edges` lying wholly inside [lo, hi]."""
    idx = [k for k in range(len(edges) - 1) if edges[k] >= lo - 1e-9 and edges[k + 1] <= hi + 1e-9]
    if not idx:
        return None
    w = np.array([(R_ORG + 0.5 * (edges[k] + edges[k + 1])) * (edges[k + 1] - edges[k]) for k in idx])
    return (vals[:, idx] * w).sum(1) / w.sum()


def thirds(vals, edges, lo, hi, n_parts=3):
    """Up to n_parts area-weighted means of the bins wholly inside [lo, hi], cutting the bin list into parts."""
    idx = [k for k in range(len(edges) - 1) if edges[k] >= lo - 1e-9 and edges[k + 1] <= hi + 1e-9]
    if not idx:
        return []
    parts = [p for p in np.array_split(np.array(idx), min(n_parts, len(idx))) if len(p)]
    out = []
    for p in parts:
        w = np.array([(R_ORG + 0.5 * (edges[k] + edges[k + 1])) * (edges[k + 1] - edges[k]) for k in p])
        out.append((vals[:, p] * w).sum(1) / w.sum())
    return out


def build(d, w):
    vs, us = d["v_shells"], d["u_shells"][:, :7]
    E = wmean(vs, RADII, w, w + 24)
    cols = thirds(vs, RADII, 0, w) + thirds(us, U_EDGES, 0, min(w, 64))
    return E, (np.column_stack(cols) if cols else np.zeros((len(E), 0)))


def task(args):
    name, w, kind, Ii, Ei, flux, band, n_null, seed, r2 = args
    Ii, Ei = zscore(Ii)[:, 0], zscore(Ei)[:, 0]
    r = np.random.default_rng(seed + 31 * w + (7 if kind == "noise" else 11 if kind == "ecopy" else 0))
    if kind == "real":
        Z = np.column_stack([flux, band])
    elif kind == "noise":
        Z = np.column_stack([flux] + [r.permutation(band[:, j]) for j in range(band.shape[1])])
    else:
        copy = Ei + r.normal(size=len(Ei)) * np.sqrt((1 - r2) / r2)
        pad = [r.permutation(band[:, j]) for j in range(max(band.shape[1] - 1, 0))]
        Z = np.column_stack([flux, copy] + pad)
    Z = zscore(Z)
    t0 = time.time()
    obs = cmi_ksg(Ii, Ei, Z)
    nc = cluster_null(Ii, Ei, Z, n_null, seed)
    nl = local_null(Ii, Ei, Z, n_null, seed)
    return (name, w, kind), dict(obs=obs, null_cluster=nc, null_local=nl, dim=Z.shape[1], secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--n-sub", type=int, default=8000)
    ap.add_argument("--n-null", type=int, default=100)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tag", default="", help="suffix for the cache and output file names")
    ap.add_argument("--only-all", action="store_true", help="skip the isolated subset")
    a = ap.parse_args()
    if a.quick:
        a.n_sub, a.n_null = 1500, 10
    d = np.load(a.data)
    I, fu, fv, ms = d["I"], d["fu"], d["fv"], d["merge_size"]
    built = {w: build(d, w) for w in WIDTHS}
    good = np.isfinite(I) & np.isfinite(fu) & np.isfinite(fv)
    for w, (E, B) in built.items():
        good &= np.isfinite(E) & np.isfinite(B).all(1)
    print(f"{good.sum():,} organism-instances with every variable finite at every width; isolated {(good & (ms == 1)).sum():,}")

    rng = np.random.default_rng(a.seed)
    subsets = {}
    for name, mask in (("all", good), ("isolated", good & (ms == 1))):
        if a.only_all and name == "isolated":
            continue
        idx = np.where(mask)[0]
        subsets[name] = rng.choice(idx, min(a.n_sub, len(idx)), replace=False)

    cache_file = CACHE.replace(".pkl", a.tag + ".pkl")
    cache = pickle.load(open(cache_file, "rb")) if os.path.exists(cache_file) else {}
    jobs = []
    for name, idx in subsets.items():
        flux = np.column_stack([fu[idx], fv[idx]])
        for w in WIDTHS:
            E, B = built[w]
            Ei, Bi = E[idx], B[idx]
            for kind in ("real", "noise", "ecopy"):
                if kind != "real" and Bi.shape[1] == 0:
                    continue
                key = (name, w, kind, a.n_sub, a.n_null, a.seed)
                if key in cache:
                    continue
                r2 = None
                if kind == "ecopy":
                    X = np.column_stack([np.ones(len(idx)), flux, Bi])
                    Ez = (Ei - Ei.mean()) / Ei.std()
                    r2 = max(1 - (Ez - X @ np.linalg.lstsq(X, Ez, rcond=None)[0]).var(), 0.02)
                jobs.append((name, w, kind, I[idx], Ei, flux, Bi, a.n_null, a.seed, r2))
    print(f"{len(jobs)} conditions to run, {a.jobs} workers")
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for (name, w, kind), res in ex.map(task, jobs):
            cache[(name, w, kind, a.n_sub, a.n_null, a.seed)] = res
            pickle.dump(cache, open(cache_file, "wb"))
            print(f"  done {name} w={w} {kind}  obs={res['obs']:.4f}  ({time.time()-t0:.0f}s total)", flush=True)

    print("\nCMI I(I; E_w | Z_w), nats; excess = observed - null mean; SD in null SDs")
    print(f"{'subset':9s} {'w':>3s} {'cond':6s} {'dim':>3s} {'obs':>7s} | {'cluster: excess':>15s} {'SD':>5s} | {'local: excess':>13s} {'SD':>5s}")
    out = {}
    for name in subsets:
        for w in WIDTHS:
            for kind in ("real", "noise", "ecopy"):
                key = (name, w, kind, a.n_sub, a.n_null, a.seed)
                if key not in cache:
                    continue
                r = cache[key]
                line = f"{name:9s} {w:3d} {kind:6s} {r['dim']:3d} {r['obs']:7.4f} |"
                for nk in ("null_cluster", "null_local"):
                    m, s = r[nk].mean(), r[nk].std()
                    line += f" {r['obs']-m:15.4f} {(r['obs']-m)/s:5.1f} |" if nk == "null_cluster" else f" {r['obs']-m:13.4f} {(r['obs']-m)/s:5.1f}"
                print(line)
                out[f"{name}_w{w}_{kind}_obs"] = r["obs"]
                out[f"{name}_w{w}_{kind}_null_cluster"] = r["null_cluster"]
                out[f"{name}_w{w}_{kind}_null_local"] = r["null_local"]
    np.savez(f"run_band_sweep{a.tag}_seed{a.seed}.npz", **out)
    print(f"\nwrote run_band_sweep{a.tag}_seed{a.seed}.npz")


if __name__ == "__main__":
    main()
