"""
Does within-frame clustering inflate the nonlinear CMI result?

The headline CMI (flux as blanket) is computed on random N=8000 subsamples of organism-instances.  Instances from the same
frame share one field realization and are spatially coupled, so they are not independent draws.  This script repeats the
CMI on subsamples built to remove that coupling, and compares each with an ordinary random subsample of the SAME size N,
so that the only thing that changes is how the instances were chosen.

Schemes (the CMI is I(I; E | Phi_u, Phi_v), all variables z-scored, Frenzel-Pompe k_NN = 5; null = local permutation among
10 nearest neighbours in flux space, as in the paper):
  random     ordinary random subsample, size matched to the scheme it is compared with (the paper's procedure)
  per_frame  at most ONE organism per frame (ensemble: per realization-and-sample; master: per frame)
  per_real   at most ONE organism per realization (ensemble only; also removes the three samples of a realization)
  thinned    within each frame keep only organisms whose centroids are at least --min-sep px apart (periodic domain),
             then draw N from the pooled result.  Default 200 px is about 10 organism radii.

Usage (from local_package):
    python3 run_cmi_independent_frames.py --quick
    python3 run_cmi_independent_frames.py --data ensemble:1000 --jobs 12
    python3 run_cmi_independent_frames.py --data master --jobs 12
Ensemble data are read from ensemble_merged_seed<BASE>_worker*.npz, master data from master_trajectory_progress.npz.
"""
import argparse
import glob
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from conditional_mi import cmi_ksg
from run_u_enriched_cmi import zscore, local_null

L_PX = 1024.0


def load(spec):
    if spec == "master":
        d = np.load("master_trajectory_progress.npz")
        frame = d["frame_id"].astype(np.int64)
        group = frame.copy()
        out = dict(I=d["I"], E=d["E"], fu=d["fu"], fv=d["fv"], cy=d["cy"], cx=d["cx"], frame=frame, group=group)
    else:
        base = spec.split(":")[1]
        files = sorted(glob.glob(f"ensemble_merged_seed{base}_worker*.npz"))
        if not files:
            raise SystemExit(f"no files ensemble_merged_seed{base}_worker*.npz in this folder")
        ds = [np.load(f) for f in files]
        g = lambda k: np.concatenate([x[k] for x in ds])
        rid, sidx = g("realization_id").astype(np.int64), g("sample_idx").astype(np.int64)
        out = dict(I=g("I"), E=g("E"), fu=g("fu"), fv=g("fv"), cy=g("cy"), cx=g("cx"), frame=rid * 10 + sidx, group=rid)
    ok = np.isfinite(out["I"]) & np.isfinite(out["E"]) & np.isfinite(out["fu"]) & np.isfinite(out["fv"])
    return {k: v[ok] for k, v in out.items()}


def one_per(labels, rng):
    order = rng.permutation(len(labels))
    _, first = np.unique(labels[order], return_index=True)
    return order[first]


def thinned_pool(d, rng, min_sep):
    keep = []
    order = rng.permutation(len(d["frame"]))
    by_frame = {}
    for i in order:
        by_frame.setdefault(d["frame"][i], []).append(i)
    for idxs in by_frame.values():
        acc = []
        for i in idxs:
            if acc:
                dy = np.abs(d["cy"][acc] - d["cy"][i]); dy = np.minimum(dy, L_PX - dy)
                dx = np.abs(d["cx"][acc] - d["cx"][i]); dx = np.minimum(dx, L_PX - dx)
                if np.hypot(dy, dx).min() < min_sep:
                    continue
            acc.append(i)
        keep.extend(acc)
    return np.array(keep)


def make_index(d, scheme, n, rng, min_sep):
    if scheme == "per_frame":
        return one_per(d["frame"], rng)
    if scheme == "per_real":
        return one_per(d["group"], rng)
    if scheme == "thinned":
        pool = thinned_pool(d, rng, min_sep)
        return rng.choice(pool, min(n, len(pool)), replace=False), len(pool)
    return rng.choice(len(d["I"]), n, replace=False)


def task(args):
    d, scheme, n, rep, n_null, min_sep = args
    rng = np.random.default_rng(1000 * rep + sum(map(ord, scheme)))
    pool = None
    idx = make_index(d, scheme, n, rng, min_sep)
    if isinstance(idx, tuple):
        idx, pool = idx
    Ii, Ei = zscore(d["I"][idx])[:, 0], zscore(d["E"][idx])[:, 0]
    Z = zscore(np.column_stack([d["fu"][idx], d["fv"][idx]]))
    t0 = time.time()
    obs = cmi_ksg(Ii, Ei, Z)
    nl = local_null(Ii, Ei, Z, n_null, rep)
    return scheme, len(idx), rep, obs, nl.mean(), nl.std(), pool, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ensemble:1000", help="ensemble:<seed base> or master")
    ap.add_argument("--n", type=int, default=8000, help="subsample size for the random and thinned schemes")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-null", type=int, default=50)
    ap.add_argument("--min-sep", type=float, default=200.0)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.reps, a.n_null = 1500, 1, 5
    d = load(a.data)
    nfr, ngr = len(np.unique(d["frame"])), len(np.unique(d["group"]))
    print(f"{a.data}: {len(d['I']):,} usable instances, {nfr:,} frames, {ngr:,} groups (realizations for an ensemble)")
    ensemble = a.data != "master"
    rng0 = np.random.default_rng(0)
    n_pf = len(one_per(d["frame"], rng0))
    n_pr = len(one_per(d["group"], rng0))
    plan = [("random", a.n, "random, N=%d" % a.n), ("thinned", a.n, "thinned (>= %.0f px apart), N=%d" % (a.min_sep, a.n)),
            ("random_pf", n_pf, "random, N=%d" % n_pf), ("per_frame", n_pf, "one per frame, N=%d" % n_pf)]
    if ensemble:
        plan += [("random_pr", n_pr, "random, N=%d" % n_pr), ("per_real", n_pr, "one per realization, N=%d" % n_pr)]
    tasks = []
    for key, n, _ in plan:
        sch = key.split("_")[0] if key.startswith("random") else key
        for rep in range(a.reps):
            tasks.append(((d, sch, n, rep, a.n_null, a.min_sep), key))
    print(f"{len(tasks)} runs, {a.n_null} null trials each, {a.jobs} workers")
    res = {k: [] for k, _, _ in plan}
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        futs = [(ex.submit(task, t), key) for t, key in tasks]
        for f, key in futs:
            r = f.result()
            res[key].append(r)
            print(f"  done {key:10s} rep {r[2]}  N={r[1]:6d}  CMI {r[3]:.4f}  null {r[4]:.4f} +/- {r[5]:.4f}  "
                  f"({(r[3]-r[4])/r[5]:.1f} SD)  [{time.time()-t0:.0f}s]", flush=True)
    print("\nSummary (mean over reps; null = local permutation, %d trials)" % a.n_null)
    print(f"{'scheme':38s} {'N':>6s} {'CMI':>8s} {'null':>8s} {'null SD':>8s} {'excess SD':>10s} {'CMI-null':>9s}")
    for key, n, label in plan:
        rows = res[key]
        obs = np.mean([r[3] for r in rows]); nm = np.mean([r[4] for r in rows]); ns = np.mean([r[5] for r in rows])
        N = int(np.mean([r[1] for r in rows]))
        print(f"{label:38s} {N:6d} {obs:8.4f} {nm:8.4f} {ns:8.4f} {(obs-nm)/ns:10.1f} {obs-nm:9.4f}")
        if key == "thinned":
            print(f"{'':38s} (pool of organisms >= {a.min_sep:.0f} px apart within a frame: {rows[0][6]:,})")
    print("\nIf clustering inflated the dependence, the 'one per ...' and 'thinned' rows would sit clearly below the random row")
    print("of the same N.  Rows of different N are compared only against their own random row.")


if __name__ == "__main__":
    main()
