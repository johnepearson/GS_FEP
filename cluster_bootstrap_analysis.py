"""
Cluster-robust bootstrap for the main partial-correlation test,
addressing VES-004 (pseudoreplication): resamples whole clusters
(frames, for the single-trajectory data; realizations, for the
ensemble data) with replacement, rather than individual
organism-instances, giving a confidence interval that properly
accounts for within-cluster correlation instead of treating every
organism-instance as independent.

Usage:
    python cluster_bootstrap_analysis.py clustered_single_trajectory_progress.npz --cluster-col frame_id
    python cluster_bootstrap_analysis.py ensemble_clustered_merged.npz --cluster-col realization_id

For the ensemble case with multiple workers, first merge them (adjust
merge_ensemble_results.py, or a short ad hoc script) into one file
with I, E, fu, fv, realization_id all concatenated, before running this.
"""
import argparse
import numpy as np


def residualize(y, B):
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def partial_corr(I, E, fu, fv):
    N = len(I)
    B = np.column_stack([np.ones(N), fu, fv])
    return np.corrcoef(residualize(I, B), residualize(E, B))[0, 1]


def cluster_bootstrap(I, E, fu, fv, cluster_id, n_boot=2000, seed=0):
    unique_clusters = np.unique(cluster_id)
    n_clusters = len(unique_clusters)
    rows_by_cluster = [np.where(cluster_id == c)[0] for c in unique_clusters]

    rng = np.random.default_rng(seed)
    boot_r = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.integers(0, n_clusters, n_clusters)
        idx = np.concatenate([rows_by_cluster[c] for c in chosen])
        boot_r[b] = partial_corr(I[idx], E[idx], fu[idx], fv[idx])
    return boot_r, n_clusters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="npz file with I, E, fu, fv, and a cluster-id column")
    ap.add_argument("--cluster-col", required=True, help="name of the cluster-id array in the file")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    d = np.load(args.file)
    I_all, E_all, fu_all, fv_all = d["I"], d["E"], d["fu"], d["fv"]
    cluster_all = d[args.cluster_col]

    good = np.isfinite(I_all) & np.isfinite(E_all) & np.isfinite(fu_all) & np.isfinite(fv_all)
    I, E, fu, fv, cluster_id = I_all[good], E_all[good], fu_all[good], fv_all[good], cluster_all[good]
    N = len(I)

    r_observed = partial_corr(I, E, fu, fv)
    se_naive = 1 / np.sqrt(N - 3)
    z = 0.5 * np.log((1 + r_observed) / (1 - r_observed))
    lo_naive, hi_naive = np.tanh(z - 1.96 * se_naive), np.tanh(z + 1.96 * se_naive)

    print(f"N organism-instances: {N}")
    print(f"N clusters ({args.cluster_col}): {len(np.unique(cluster_id))}")
    print(f"observed partial correlation: r = {r_observed:.4f}")
    print(f"naive 95% CI (treats every instance as independent): [{lo_naive:.4f}, {hi_naive:.4f}]")
    print()
    print(f"running cluster bootstrap ({args.n_boot} resamples)...")

    boot_r, n_clusters = cluster_bootstrap(I, E, fu, fv, cluster_id, n_boot=args.n_boot)
    lo_boot, hi_boot = np.percentile(boot_r, [2.5, 97.5])

    print()
    print(f"cluster-bootstrap 95% CI (percentile method, {n_clusters} clusters): [{lo_boot:.4f}, {hi_boot:.4f}]")
    print(f"bootstrap distribution: mean={boot_r.mean():.4f}  std={boot_r.std():.4f}")
    print()
    print("Compare the two CIs above: if the cluster-bootstrap interval is only modestly")
    print("wider and still excludes zero by a comfortable margin, the naive analysis's")
    print("qualitative conclusion is not threatened by clustering, even though its precision")
    print("was likely overstated.")

    np.savez(args.file.replace(".npz", "_bootstrap_result.npz"),
             r_observed=r_observed, boot_r=boot_r, n_clusters=n_clusters, N=N)


if __name__ == "__main__":
    main()
