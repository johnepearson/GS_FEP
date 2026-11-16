"""
Pooled nonlinear collider check on the full population (supplement, "Boundary-concentration collider test").

For the full population of merge-corrected organism-instances (no split by merge size) this computes, on N=8000
subsamples, the unconditioned nonlinear MI I(I;E), the flux-conditioned CMI and the boundary-concentration-conditioned CMI,
each z-scored, each against a local-permutation null (300 trials).  The subsample is redrawn for each seed and the
three quantities are printed per seed and as a mean over seeds, because the N=8000 estimate of I(I;E) varies by about
+-0.012 nats between subsamples.

Input: master_trajectory_progress.npz (the single-trajectory dataset, merge-corrected segmentation with minimum peak
distance 24; fields I, E, fu, fv, boundary_u, boundary_v).

Usage (from local_package):
    python3 analyze_pooled_collider_cmi.py --seeds 0 1 2 --data master_trajectory_progress.npz   # about 30 minutes on one core
    python3 analyze_pooled_collider_cmi.py --self-test          # synthetic check of the estimator
"""
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma

K_MI = 5          # neighbors for the KSG / Frenzel-Pompe estimator itself
K_PERM = 10       # neighbors (in conditioning-variable space) used to build
                  # the local-permutation null
N_TRIALS = 300    # null trials
SEED = 0


# ---------------------------------------------------------------------
# KSG (Kraskov-Stogbauer-Grassberger) / Frenzel-Pompe estimators
# ---------------------------------------------------------------------

def _count_within(tree, points, radii):
    """For each point i, count neighbors (excluding self) strictly within
    radii[i] (open ball), Chebyshev/max-norm, as KSG's algorithm requires."""
    counts = np.empty(len(points), dtype=int)
    for i in range(len(points)):
        # query_ball_point with the (shrunk) radius, open ball => subtract self
        idx = tree.query_ball_point(points[i], r=radii[i] - 1e-12, p=np.inf)
        counts[i] = len(idx) - 1 if len(idx) > 0 else 0
        if counts[i] < 0:
            counts[i] = 0
    return counts


def mutual_information_ksg(X, Y, k=K_MI):
    """KSG estimator (2004), algorithm 1, max-norm, for continuous I(X;Y)."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    Z = np.hstack([X, Y])
    n = Z.shape[0]

    tree_z = cKDTree(Z)
    # k+1 because the point itself is included at distance 0
    dists, _ = tree_z.query(Z, k=k + 1, p=np.inf)
    eps = dists[:, -1]  # distance to k-th neighbor, excluding self correctly
                         # (query includes self as the 0th neighbor)

    tree_x = cKDTree(X)
    tree_y = cKDTree(Y)
    nx = _count_within(tree_x, X, eps)
    ny = _count_within(tree_y, Y, eps)

    mi = digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n)
    return max(mi, 0.0)


def conditional_mi_fp(X, Y, Z, k=K_MI):
    """Frenzel-Pompe (2007) generalization of KSG to the conditional case,
    I(X;Y|Z), max-norm throughout."""
    X = np.atleast_2d(X.T).T if X.ndim == 1 else X
    Y = np.atleast_2d(Y.T).T if Y.ndim == 1 else Y
    Z = np.atleast_2d(Z.T).T if Z.ndim == 1 else Z
    XYZ = np.hstack([X, Y, Z])
    n = XYZ.shape[0]

    tree_xyz = cKDTree(XYZ)
    dists, _ = tree_xyz.query(XYZ, k=k + 1, p=np.inf)
    eps = dists[:, -1]

    tree_xz = cKDTree(np.hstack([X, Z]))
    tree_yz = cKDTree(np.hstack([Y, Z]))
    tree_z = cKDTree(Z)

    nxz = _count_within(tree_xz, np.hstack([X, Z]), eps)
    nyz = _count_within(tree_yz, np.hstack([Y, Z]), eps)
    nz = _count_within(tree_z, Z, eps)

    cmi = (digamma(k)
           - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1)))
    return max(cmi, 0.0)


# ---------------------------------------------------------------------
# Local-permutation null (Runge 2018 CMI-knn style): shuffle Y only among
# points that are mutual near-neighbors in Z-space, approximately
# preserving p(Y|Z) while destroying any extra X-Y dependence beyond Z.
# ---------------------------------------------------------------------

def local_permutation(n, Z, k_perm=K_PERM, rng=None):
    rng = rng or np.random.default_rng()
    tree_z = cKDTree(Z)
    _, neighbor_idx = tree_z.query(Z, k=k_perm + 1, p=np.inf)  # includes self
    perm = np.arange(n)
    # Many random local swaps: pick a point, pick one of its z-neighbors,
    # swap their current assignment in perm. This keeps every assignment
    # anchored to a nearby Z, unlike a global shuffle.
    n_swaps = 4 * n
    for _ in range(n_swaps):
        i = rng.integers(n)
        cand = neighbor_idx[i]
        cand = cand[cand != i]
        if len(cand) == 0:
            continue
        j = cand[rng.integers(len(cand))]
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def cmi_with_null(X, Y, Z, k=K_MI, k_perm=K_PERM, n_trials=N_TRIALS, seed=SEED):
    observed = conditional_mi_fp(X, Y, Z, k=k)
    if n_trials <= 0:
        return observed, float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    null_vals = np.empty(n_trials)
    n = len(X)
    for t in range(n_trials):
        perm = local_permutation(n, Z, k_perm=k_perm, rng=rng)
        null_vals[t] = conditional_mi_fp(X, Y[perm], Z, k=k)
    null_mean, null_std = null_vals.mean(), null_vals.std()
    null_sds = (observed - null_mean) / null_std if null_std > 0 else np.inf
    return observed, null_mean, null_std, null_sds


def zscore(a):
    a = np.asarray(a, dtype=float)
    return (a - a.mean()) / a.std()


# ---------------------------------------------------------------------
# Self-test on synthetic data with a known, planted dependence structure
# ---------------------------------------------------------------------

def self_test():
    print("Running self-test on synthetic data (no real data needed)...")
    rng = np.random.default_rng(1)
    n = 4000

    # Planted structure: I and E both driven by a shared latent "flux" B,
    # plus independent noise -- so conditioning on B should screen most of
    # the I-E dependence (NOT a collider), while a second variable W built
    # as a function of I and E together (B2 = I + E + noise) is, by
    # construction, a collider: conditioning on it should *increase*
    # apparent I-E dependence, same signature flux is diagnosed as having
    # in the real data.
    B = rng.normal(size=n)
    I = 0.6 * B + rng.normal(scale=0.5, size=n)
    E = 0.6 * B + rng.normal(scale=0.5, size=n)
    collider = I + E + rng.normal(scale=0.3, size=n)

    I_z, E_z, B_z, C_z = zscore(I), zscore(E), zscore(B), zscore(collider)

    mi_unconditioned = mutual_information_ksg(I_z, E_z)
    cmi_B, nm_B, ns_B, nsd_B = cmi_with_null(I_z, E_z, B_z.reshape(-1, 1),
                                              n_trials=60)
    cmi_C, nm_C, ns_C, nsd_C = cmi_with_null(I_z, E_z, C_z.reshape(-1, 1),
                                              n_trials=60)

    print(f"  unconditioned I(I;E)          = {mi_unconditioned:.4f} nats")
    print(f"  conditioned on real mediator B = {cmi_B:.4f} nats "
          f"(null {nm_B:.4f}+-{ns_B:.4f}, {nsd_B:.1f} null-SDs) "
          f"-> screened {1 - cmi_B / mi_unconditioned:+.1%}")
    print(f"  conditioned on collider C      = {cmi_C:.4f} nats "
          f"(null {nm_C:.4f}+-{ns_C:.4f}, {nsd_C:.1f} null-SDs) "
          f"-> screened {1 - cmi_C / mi_unconditioned:+.1%}")
    print()
    ok = (cmi_B < mi_unconditioned) and (cmi_C > mi_unconditioned * 0.9)
    print("Self-test sanity check:",
          "PASS -- mediator screens, collider does not" if ok else
          "CHECK MANUALLY -- pattern not as expected, inspect numbers above")


# ---------------------------------------------------------------------
# Main: pooled real-data run
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--field-I", default="I")
    ap.add_argument("--field-E", default="E")
    ap.add_argument("--field-fu", default="fu")
    ap.add_argument("--field-fv", default="fv")
    ap.add_argument("--field-ub", default="boundary_u")
    ap.add_argument("--field-vb", default="boundary_v")
    ap.add_argument("--n-sub", type=int, default=8000,
                     help="subsample size for the CMI estimator (full population is too slow "
                          "pooled population is too slow "
                          "for O(n log n)-per-trial kNN x 300 trials x "
                          "3 conditioning variables; 8000 matches the "
                          "clean-subset convention in Table 4)")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-seeds", type=int, default=0,
                    help="use seeds 0..n-1 (overrides --seeds); build/initialize.tex: CmiSubsamples")
    ap.add_argument("--null-seeds", type=int, default=None,
                    help="only the first this-many subsamples also get the local-permutation null "
                         "(default: all); build/initialize.tex: CmiNullSubsamples")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    d = np.load(args.data)
    print("Available fields in", args.data, ":", list(d.keys()))

    def get(field):
        if field not in d:
            raise SystemExit(
                f"Field '{field}' not found in {args.data}. "
                f"Available fields: {list(d.keys())}. "
                f"Pass the correct name with --field-... (see --help)."
            )
        return d[field]

    I = get(args.field_I).astype(float)
    E = get(args.field_E).astype(float)
    fu = get(args.field_fu).astype(float)
    fv = get(args.field_fv).astype(float)
    ub = get(args.field_ub).astype(float)
    vb = get(args.field_vb).astype(float)

    # pooled population: no merge_size split at all, just the standard
    # finite-value filter already used to build this dataset's headline N.
    finite = (np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
              & np.isfinite(ub) & np.isfinite(vb))
    I, E, fu, fv, ub, vb = (a[finite] for a in (I, E, fu, fv, ub, vb))
    n_total = len(I)
    print(f"pooled population after finite-value filter: N={n_total}")
    I0, E0, fu0, fv0, ub0, vb0 = I, E, fu, fv, ub, vb
    if args.n_seeds > 0:
        args.seeds = list(range(args.n_seeds))
    results = []
    for i, seed in enumerate(args.seeds):
        with_null = args.null_seeds is None or i < args.null_seeds
        results.append(run_seed(args, seed, I0, E0, fu0, fv0, ub0, vb0, n_total, with_null))
    r = np.array(results)
    write_results(args, r, n_total)
    if len(r) > 1:
        print("\n=== mean over seeds " + " ".join(map(str, args.seeds)) + " ===")
        print(f"I(I;E) = {r[:,0].mean():.4f} nats")
        print(f"Flux:      CMI={r[:,1].mean():.4f}  null-SDs {np.nanmin(r[:,2]) if np.isfinite(r[:,2]).any() else float('nan'):.1f}-{np.nanmax(r[:,2]) if np.isfinite(r[:,2]).any() else float('nan'):.1f}  "
              f"screened={(1 - r[:,1].mean()/r[:,0].mean()):+.1%}")
        print(f"Boundary:  CMI={r[:,3].mean():.4f}  null-SDs {np.nanmin(r[:,4]) if np.isfinite(r[:,4]).any() else float('nan'):.1f}-{np.nanmax(r[:,4]) if np.isfinite(r[:,4]).any() else float('nan'):.1f}  "
              f"screened={(1 - r[:,3].mean()/r[:,0].mean()):+.1%}")


def write_results(args, r, n_total):
    """Macros for the paper: means over the subsamples, standard errors, ranges (columns of r: unconditioned MI,
    flux CMI, flux null-SDs, boundary CMI, boundary null-SDs; null-SDs are nan where no null was run)."""
    from texout import num, write_macros
    k = len(r)
    mi, fl, bd = r[:, 0], r[:, 1], r[:, 3]
    se = lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else float("nan")
    up = fl / mi - 1.0                      # per-subsample change from conditioning on flux
    scr = 1.0 - bd / mi                     # per-subsample fraction removed by the boundary concentration
    nf = r[~np.isnan(r[:, 2]), 2]
    nb = r[~np.isnan(r[:, 4]), 4]
    items = [
        ("PooledSubsamples", num(k, 0)), ("PooledNSub", num(args.n_sub, 0)), ("PooledTrials", num(args.n_trials, 0)),
        ("PooledPopulation", num(n_total, 0)),
        ("PooledMI", num(mi.mean(), 5)), ("PooledMISE", num(se(mi), 5)),
        ("PooledFluxCMI", num(fl.mean(), 5)), ("PooledFluxCMISE", num(se(fl), 5)),
        ("PooledFluxRise", num(100 * (fl.mean() / mi.mean() - 1), 2)),
        ("PooledFluxHigher", num(int((fl > mi).sum()), 0)),
        ("PooledFluxRiseMin", num(100 * up.min(), 1)), ("PooledFluxRiseMax", num(100 * up.max(), 1)),
        ("PooledBoundaryCMI", num(bd.mean(), 5)), ("PooledBoundaryCMISE", num(se(bd), 5)),
        ("PooledBoundaryScreened", num(100 * (1 - bd.mean() / mi.mean()), 2)),
        ("PooledBoundaryScreenedMean", num(100 * scr.mean(), 1)), ("PooledBoundaryScreenedSD", num(100 * scr.std(ddof=1) if k > 1 else float("nan"), 1)),
        ("PooledNullSubsamples", num(len(nf), 0)),
    ]
    if len(nf):
        items += [("PooledFluxSDLo", num(nf.min(), 1)), ("PooledFluxSDHi", num(nf.max(), 1)),
                  ("PooledBoundarySDLo", num(nb.min(), 1)), ("PooledBoundarySDHi", num(nb.max(), 1)),
                  ("PooledSDLo", num(min(nf.min(), nb.min()), 0)), ("PooledSDHi", num(max(nf.max(), nb.max()), 0))]
    for i, name in enumerate(("A", "B", "C")):          # the flux CMI of the first three (null-tested) subsamples
        if i < k:
            items.append(("PooledFluxCMI" + name, num(fl[i], 3)))
    write_macros("results/pooledcmi.tex", "analyze_pooled_collider_cmi.py", items)


def run_seed(args, seed, I, E, fu, fv, ub, vb, n_total, with_null=True):
    n_trials = args.n_trials if with_null else 0
    print(f"\n########## seed {seed} ##########")

    rng = np.random.default_rng(seed)
    if n_total > args.n_sub:
        sub = rng.choice(n_total, size=args.n_sub, replace=False)
        I, E, fu, fv, ub, vb = (a[sub] for a in (I, E, fu, fv, ub, vb))
    n = len(I)
    print(f"CMI subsample: N={n}")

    I_z, E_z = zscore(I), zscore(E)
    flux_z = np.column_stack([zscore(fu), zscore(fv)])
    bc_z = np.column_stack([zscore(ub), zscore(vb)])

    print("\nComputing unconditioned nonlinear MI I(I;E) ...")
    mi0 = mutual_information_ksg(I_z, E_z)
    print(f"  I(I;E) = {mi0:.4f} nats")

    print(f"\nComputing flux-conditioned CMI ({n_trials} null trials)...")
    cmi_flux, nm_f, ns_f, nsd_f = cmi_with_null(I_z, E_z, flux_z,
                                                 n_trials=n_trials,
                                                 seed=seed)
    screened_flux = (1 - cmi_flux / mi0) if mi0 > 0 else float("nan")
    print(f"  CMI(I;E|flux)      = {cmi_flux:.4f} nats   "
          f"null {nm_f:.4f} (std {ns_f:.4f})   {nsd_f:.1f} null-SDs   "
          f"nonlinear screened: {screened_flux:+.1%}")

    print(f"\nComputing boundary-concentration-conditioned CMI "
          f"({n_trials} null trials)...")
    cmi_bc, nm_b, ns_b, nsd_b = cmi_with_null(I_z, E_z, bc_z,
                                               n_trials=n_trials,
                                               seed=seed + 1)
    screened_bc = (1 - cmi_bc / mi0) if mi0 > 0 else float("nan")
    print(f"  CMI(I;E|boundary)  = {cmi_bc:.4f} nats   "
          f"null {nm_b:.4f} (std {ns_b:.4f})   {nsd_b:.1f} null-SDs   "
          f"nonlinear screened: {screened_bc:+.1%}")

    print("\n--- summary ---")
    print(f"Pooled population, N={n} (subsampled from {n_total})")
    print(f"Unconditioned I(I;E) = {mi0:.4f} nats")
    print(f"Flux:      CMI={cmi_flux:.4f}  null={nm_f:.4f}({ns_f:.4f})  "
          f"{nsd_f:.1f} null-SDs  screened={screened_flux:+.1%}")
    print(f"Boundary:  CMI={cmi_bc:.4f}  null={nm_b:.4f}({ns_b:.4f})  "
          f"{nsd_b:.1f} null-SDs  screened={screened_bc:+.1%}")
    return mi0, cmi_flux, nsd_f, cmi_bc, nsd_b


if __name__ == "__main__":
    main()
