"""
Post-processing for run_ensemble_merged.py's output (main text §3.3.1, Table 1). Merges all matching worker npz
files for one ensemble (--seed-base 1000, 9000 or 20000), then reports:

  - pooled partial correlation (plain joint-OLS, on the full unstratified population; see partial_corr_utils.py)
  - per-realization partial-correlation distribution (mean, std, percentiles, fraction sharing the pooled
    estimate's sign)
  - realization-cluster bootstrap 95% CI on the pooled estimate
  - merge_size distribution, as a sanity check against the single-trajectory merge rate

Worker files are named ensemble_merged_seed{seed_base}_worker{w}.npz, as written by run_ensemble_merged.py.

Usage:
    python analyze_ensemble_merged.py --n-workers 12 --seed-base 9000
    python analyze_ensemble_merged.py --n-workers 12 --seed-base 1000
"""
import argparse
import numpy as np

from cluster_bootstrap_analysis import partial_corr, cluster_bootstrap

MIN_N = 20  # minimum organism-instances in a realization to compute its
            # own partial correlation; smaller realizations are pooled
            # into the sanity-check number but excluded from the
            # per-realization distribution as too noisy individually.

def load_and_merge(n_workers, seed_base):
    I_list, E_list, fu_list, fv_list, rid_list, ms_list = [], [], [], [], [], []
    n_realizations_total = 0
    # run_ensemble_merged.py writes the seed-tagged name for every seed base.
    pattern = f"ensemble_merged_seed{seed_base}_worker" + "{w}.npz"
    for w in range(n_workers):
        path = pattern.format(w=w)
        try:
            d = np.load(path)
        except FileNotFoundError:
            print(f"  WARNING: {path} not found, skipping")
            continue
        I_list.append(d["I"]); E_list.append(d["E"])
        fu_list.append(d["fu"]); fv_list.append(d["fv"])
        rid_list.append(d["realization_id"])
        ms_list.append(d["merge_size"])
        n_seeds = len(d["seeds_done"])
        n_realizations_total += n_seeds
        print(f"  {path}: {n_seeds} realizations, {len(d['I'])} organism-instances")

    I = np.concatenate(I_list); E = np.concatenate(E_list)
    fu = np.concatenate(fu_list); fv = np.concatenate(fv_list)
    rid = np.concatenate(rid_list)
    ms = np.concatenate(ms_list)

    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    n_dropped = (~good).sum()
    if n_dropped:
        print(f"  [{n_dropped} non-finite organism-instances dropped]")
    return I[good], E[good], fu[good], fv[good], rid[good], ms[good], n_realizations_total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-workers", type=int, required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--min-n", type=int, default=MIN_N)
    ap.add_argument("--seed-base", type=int, default=9000,
                     help="Which ensemble's worker files to load: 9000 (default) for "
                          "Ensemble 2, 1000 for Ensemble 1, 20000 for Ensemble 3 "
                          "(ensemble_merged_seed{base}_worker{w}.npz). Must match the "
                          "--seed-base used for the run_ensemble_merged.py run being analyzed.")
    args = ap.parse_args()

    print(f"loading and merging worker files for seed-base={args.seed_base} ...")
    I, E, fu, fv, rid, ms, n_realizations = load_and_merge(args.n_workers, args.seed_base)
    n_unique = len(np.unique(rid))
    print(f"\ntotal: {n_realizations} realizations recorded, {n_unique} with >=1 organism, "
          f"N={len(I)} organism-instances")

    print(f"\nmerge_size distribution: {np.bincount(ms)}")
    was_merged = ms > 1
    print(f"fraction that are merges of 2+ original sub-regions: {was_merged.mean():.4f}  "
          f"(single-trajectory comparison: ~0.10)")

    r_pooled = partial_corr(I, E, fu, fv)
    print(f"\npooled partial r (all realizations combined) = {r_pooled:.4f}")

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
    frac_excl_zero_sign = ((per_real_r < 0).mean() if r_pooled < 0 else (per_real_r > 0).mean())
    print(f"  fraction of realizations with the same sign as the pooled estimate: "
          f"{frac_excl_zero_sign:.1%}")

    print(f"\nrunning realization-cluster bootstrap on the pooled estimate "
          f"({args.n_boot} resamples)...")
    boot_r, n_clusters = cluster_bootstrap(I, E, fu, fv, rid, n_boot=args.n_boot)
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    print(f"  realization-cluster bootstrap 95% CI on pooled r: [{lo:.4f}, {hi:.4f}] "
          f"({n_clusters} clusters)")

    out_name = f"ensemble_merged_seed{args.seed_base}_distribution_result.npz"
    np.savez(out_name,
              r_pooled=r_pooled, per_real_r=per_real_r, per_real_n=per_real_n,
              n_realizations=n_realizations, n_unique=n_unique,
              boot_r=boot_r, lo=lo, hi=hi, n_clusters=n_clusters, merge_size=ms)
    print(f"\nsaved {out_name}")
    print("(per_real_r is the array to histogram/plot for the realization-distribution figure)")


if __name__ == "__main__":
    main()
