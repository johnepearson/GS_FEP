"""
Post-processing for run_ensemble_clustered.py's output: merges all
worker files, then computes the paper's headline (joint two-flux-
variable OLS) partial correlation ONCE PER REALIZATION, giving the
actual distribution of the headline statistic across ~1000 independent
realizations -- what the 2026-09-28 review asked for (item #4) as a
more direct, referee-facing alternative/complement to a cluster
bootstrap: rather than only a resampled CI around the pooled estimate,
this shows what the statistic looks like when computed fresh on each
independent noise-perturbed trajectory.

Reports:
  - a sanity check: the pooled (all realizations combined) partial
    correlation should be close to the paper's existing ensemble
    headline (r=-0.152, N=609,951) -- not identical, since this is a
    FRESH set of 1000 realizations with different seeds (9000+i, not
    the original 1000+i) and only 3 samples per realization rather
    than however many the original profile-collection ensemble used,
    but the same production methodology otherwise, so it should be a
    comparable draw from the same underlying distribution.
  - per-realization partial correlation: computed separately for each
    realization with at least MIN_N organism-instances (small
    realizations give a very noisy single-realization estimate and are
    excluded from the distribution, though still contribute to the
    pooled sanity-check number above). Reports mean, std, and
    percentiles of this per-realization distribution, and saves the
    raw array for plotting.
  - for completeness, also runs the existing realization-cluster
    bootstrap (cluster_bootstrap_analysis.cluster_bootstrap, treating
    realization_id as the cluster column) on the pooled data, which
    answers a related but different question (uncertainty in the
    POOLED estimate, accounting for within-realization correlation,
    rather than the spread of separately-computed per-realization
    estimates).

Usage:
    python analyze_realization_distribution.py --n-workers 4
"""
import argparse
import numpy as np

from cluster_bootstrap_analysis import partial_corr, cluster_bootstrap

MIN_N = 20  # minimum organism-instances in a realization to compute its
            # own partial correlation; smaller realizations are pooled
            # into the sanity-check number but excluded from the
            # per-realization distribution as too noisy individually.


def load_and_merge(n_workers):
    I_list, E_list, fu_list, fv_list, rid_list = [], [], [], [], []
    n_realizations_total = 0
    for w in range(n_workers):
        path = f"ensemble_clustered_worker{w}.npz"
        try:
            d = np.load(path)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found, skipping")
            continue
        I_list.append(d["I"]); E_list.append(d["E"])
        fu_list.append(d["fu"]); fv_list.append(d["fv"])
        rid_list.append(d["realization_id"])
        n_seeds = len(d["seeds_done"])
        n_realizations_total += n_seeds
        print(f"  {path}: {n_seeds} realizations, {len(d['I'])} organism-instances")

    I = np.concatenate(I_list); E = np.concatenate(E_list)
    fu = np.concatenate(fu_list); fv = np.concatenate(fv_list)
    rid = np.concatenate(rid_list)

    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    return I[good], E[good], fu[good], fv[good], rid[good], n_realizations_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    args = ap.parse_args()

    print("loading and merging worker files...")
    I, E, fu, fv, rid, n_realizations = load_and_merge(args.n_workers)
    n_unique = len(np.unique(rid))
    print(f"\ntotal: {n_realizations} realizations recorded, {n_unique} with >=1 organism, "
          f"N={len(I)} organism-instances")

    r_pooled = partial_corr(I, E, fu, fv)
    print(f"\npooled partial r (all realizations combined) = {r_pooled:.4f}  "
          f"(sanity check vs. paper's existing ensemble headline: r=-0.152, N=609,951; "
          f"not expected to match exactly -- fresh seeds, different N)")

    print(f"\ncomputing per-realization partial correlation "
          f"(realizations with >={args.min_n} organism-instances)...")
    unique_rid = np.unique(rid)
    per_real_r = []
    per_real_n = []
    n_skipped_small = 0
    for r in unique_rid:
        sel = rid == r
        n_r = sel.sum()
        if n_r < args.min_n:
            n_skipped_small += 1
            continue
        r_val = partial_corr(I[sel], E[sel], fu[sel], fv[sel])
        if np.isfinite(r_val):
            per_real_r.append(r_val)
            per_real_n.append(n_r)

    per_real_r = np.array(per_real_r)
    per_real_n = np.array(per_real_n)
    print(f"  {len(per_real_r)} realizations used (>= {args.min_n} instances each), "
          f"{n_skipped_small} skipped as too small")
    print(f"  organism-instances per used realization: min={per_real_n.min()}, "
          f"median={np.median(per_real_n):.0f}, max={per_real_n.max()}")

    print(f"\n=== distribution of the partial correlation across {len(per_real_r)} "
          f"independent realizations ===")
    print(f"  mean   = {per_real_r.mean():.4f}")
    print(f"  std    = {per_real_r.std():.4f}")
    print(f"  median = {np.median(per_real_r):.4f}")
    pcts = np.percentile(per_real_r, [2.5, 25, 50, 75, 97.5])
    print(f"  2.5/25/50/75/97.5 percentiles: {pcts[0]:.4f}, {pcts[1]:.4f}, {pcts[2]:.4f}, "
          f"{pcts[3]:.4f}, {pcts[4]:.4f}")
    frac_negative = (per_real_r < 0).mean()
    frac_excl_zero_sign = ((per_real_r < 0).mean() if r_pooled < 0 else (per_real_r > 0).mean())
    print(f"  fraction of realizations with the same sign as the pooled estimate: "
          f"{frac_excl_zero_sign:.1%}")

    print(f"\nrunning realization-cluster bootstrap on the pooled estimate "
          f"({args.n_boot} resamples, for comparison)...")
    boot_r, n_clusters = cluster_bootstrap(I, E, fu, fv, rid, n_boot=args.n_boot)
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    print(f"  realization-cluster bootstrap 95% CI on pooled r: [{lo:.4f}, {hi:.4f}] "
          f"({n_clusters} clusters)")

    np.savez("realization_distribution_result.npz",
              r_pooled=r_pooled, per_real_r=per_real_r, per_real_n=per_real_n,
              n_realizations=n_realizations, n_unique=n_unique,
              boot_r=boot_r, lo=lo, hi=hi, n_clusters=n_clusters)
    print("\nsaved realization_distribution_result.npz")
    print("(per_real_r is the array to histogram/plot for the reviewer-facing distribution figure)")


if __name__ == "__main__":
    main()
