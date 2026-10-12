"""
Is the flux-conditioned I-E dependence a genuine WITHIN-organism, moment-to-moment coupling, or a mixture (ecological-
correlation) artifact of pooling many organisms (lineages) together?  A mixture artifact would arise if different
lineages sat at different average interior, exterior and flux levels, so that the population-level dependence came
from between-lineage differences rather than from how any single organism's own states co-vary.

Uses the lineage-tracked dataset written by analyze_lifecycle_age.py (lifecycle_age_analysis.npz): every sampled frame of
every tracked organism's own lifetime, recording (lineage_id, t, age, I, E, fu, fv).

Two checks:
(1) WITHIN-LINEAGE fixed-effects partial correlation: demean I, E and the combined flux regressor (PC1 of standardized
    fu, fv) separately within each lineage, then compute the partial correlation.  This asks only whether an organism's
    own I and E, relative to its own averages, still co-vary once its own flux is controlled for, and it is blind to
    between-lineage mean differences.  A pure mixture artifact would make it collapse to about zero.

(2) Law-of-total-covariance decomposition of the flux-RESIDUALIZED I, E
    (i.e. the same quantities the headline partial correlation
    correlates) into a within-lineage share (each lineage's own
    residual covariance, pooled and weighted by lineage size) and a
    between-lineage share (covariance of each lineage's own MEAN
    residual level), mirroring Table tab:covariance_decomposition's
    construction exactly but grouped by organism identity instead of
    touching/isolated status.

Both are bootstrapped by resampling whole LINEAGES (not individual
measurements), matching analyze_lifecycle_age.py's existing convention,
since measurements from the same lineage are not independent.

Usage:
    python run_lineage_mixture_decomposition.py --data lifecycle_age_analysis.npz
"""
import argparse
import numpy as np

N_BOOT = 500
SEED = 0


def load(path):
    d = np.load(path, allow_pickle=True)
    lid, I, E, fu, fv = d["lineage_id_hash"], d["I"], d["E"], d["fu"], d["fv"]
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    return lid[good], I[good], E[good], fu[good], fv[good]


def pc1_flux(fu, fv):
    """Single combined flux regressor: PC1 of standardized (fu, fv),
    matching the paper's convention (partial_corr_utils.pc1_partial_corr)."""
    S = np.column_stack([fu, fv])
    Sc = (S - S.mean(axis=0)) / S.std(axis=0)
    U, Sv, Vt = np.linalg.svd(Sc, full_matrices=False)
    return U[:, 0] * Sv[0]


def _residualize(y, B):
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def naive_partial_corr(I, E, fu, fv):
    """Pooled partial correlation, ignoring lineage structure entirely
    -- reproduces the paper's existing headline number on this dataset
    (r=-0.196) as a sanity check before doing anything new."""
    p1 = pc1_flux(fu, fv)
    n = len(I)
    B = np.column_stack([np.ones(n), p1])
    r = np.corrcoef(_residualize(I, B), _residualize(E, B))[0, 1]
    return r


def within_lineage_partial_corr(lineage_id, I, E, fu, fv, min_size=2):
    """Lineage fixed-effects ('within') partial correlation: demean I,
    E, and the flux regressor separately within each lineage, then
    correlate the demeaned I against the demeaned E after also
    residualizing out the demeaned flux regressor. Lineages with fewer
    than min_size measurements contribute no within-group variation
    and are dropped (their demeaned values would be identically zero
    or undefined)."""
    p1 = pc1_flux(fu, fv)
    unique_lids, counts = np.unique(lineage_id, return_counts=True)
    keep_lids = set(unique_lids[counts >= min_size])
    mask = np.array([lid in keep_lids for lid in lineage_id])

    I_m, E_m, p1_m, lid_m = I[mask], E[mask], p1[mask], lineage_id[mask]

    I_dm = np.empty_like(I_m)
    E_dm = np.empty_like(E_m)
    p1_dm = np.empty_like(p1_m)
    for lid in np.unique(lid_m):
        sel = lid_m == lid
        I_dm[sel] = I_m[sel] - I_m[sel].mean()
        E_dm[sel] = E_m[sel] - E_m[sel].mean()
        p1_dm[sel] = p1_m[sel] - p1_m[sel].mean()

    n = len(I_dm)
    B = np.column_stack([p1_dm])  # no intercept needed: everything is already zero-mean per lineage
    r = np.corrcoef(_residualize(I_dm, B), _residualize(E_dm, B))[0, 1]
    n_lineages_used = len(keep_lids)
    return r, n, n_lineages_used


def within_between_covariance_decomposition(lineage_id, I, E, fu, fv):
    """Law-of-total-covariance decomposition of the FLUX-RESIDUALIZED
    I, E (Delta_I, Delta_E -- the same quantities the headline partial
    correlation itself correlates) into a within-lineage share and a
    between-lineage share, weighted by lineage size, mirroring
    Table tab:covariance_decomposition's construction."""
    p1 = pc1_flux(fu, fv)
    n = len(I)
    B = np.column_stack([np.ones(n), p1])
    dI = _residualize(I, B)
    dE = _residualize(E, B)

    unique_lids = np.unique(lineage_id)
    n_lin = len(unique_lids)
    within_cov_sum = 0.0
    lineage_means_I = np.empty(n_lin)
    lineage_means_E = np.empty(n_lin)
    lineage_weights = np.empty(n_lin)

    for k, lid in enumerate(unique_lids):
        sel = lineage_id == lid
        n_k = sel.sum()
        lineage_weights[k] = n_k
        mI, mE = dI[sel].mean(), dE[sel].mean()
        lineage_means_I[k] = mI
        lineage_means_E[k] = mE
        if n_k > 1:
            within_cov_sum += n_k * np.mean((dI[sel] - mI) * (dE[sel] - mE))

    within_cov = within_cov_sum / n  # weighted average within-lineage covariance
    grand_mean_I, grand_mean_E = dI.mean(), dE.mean()
    between_cov = np.sum(lineage_weights * (lineage_means_I - grand_mean_I)
                          * (lineage_means_E - grand_mean_E)) / n
    total_cov = np.cov(dI, dE, bias=True)[0, 1]

    # sanity check: within_cov + between_cov should equal total_cov (law of total covariance)
    check = within_cov + between_cov
    return dict(within_cov=within_cov, between_cov=between_cov, total_cov=total_cov,
                check=check, n_lineages=n_lin,
                within_share=within_cov / total_cov if total_cov != 0 else np.nan,
                between_share=between_cov / total_cov if total_cov != 0 else np.nan)


def bootstrap_by_lineage(fn, lineage_id, *arrays, n_boot=N_BOOT, seed=SEED):
    unique_lids = np.unique(lineage_id)
    n_lin = len(unique_lids)
    rows_by_lid = {lid: np.where(lineage_id == lid)[0] for lid in unique_lids}

    rng = np.random.default_rng(seed)
    results = []
    for b in range(n_boot):
        chosen = rng.choice(unique_lids, n_lin, replace=True)
        idx = np.concatenate([rows_by_lid[lid] for lid in chosen])
        # relabel resampled lineages uniquely so repeats of the same
        # lineage in one bootstrap draw are not merged into one group
        new_lid = np.concatenate([np.full(len(rows_by_lid[lid]), i)
                                   for i, lid in enumerate(chosen)])
        arrs = [a[idx] for a in arrays]
        results.append(fn(new_lid, *arrs))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="lifecycle_age_analysis.npz")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    lid, I, E, fu, fv = load(args.data)
    n_lineages = len(np.unique(lid))
    print(f"loaded N={len(I)} measurements across {n_lineages} lineages")

    r_naive = naive_partial_corr(I, E, fu, fv)
    print(f"\npooled (naive) partial r, ignoring lineage structure = {r_naive:.4f}  "
          )

    print("\n--- (1) within-lineage fixed-effects partial correlation ---")
    r_within, n_used, n_lin_used = within_lineage_partial_corr(lid, I, E, fu, fv)
    print(f"within-lineage r_partial = {r_within:.4f}  "
          f"(N={n_used} measurements from {n_lin_used} lineages with >=2 measurements each)")

    print(f"\nrunning lineage-cluster bootstrap ({args.n_boot} resamples) for within-lineage r...")
    def within_fn(new_lid, I_, E_, fu_, fv_):
        r, n, nl = within_lineage_partial_corr(new_lid, I_, E_, fu_, fv_)
        return r
    boot_within = np.array(bootstrap_by_lineage(within_fn, lid, I, E, fu, fv, n_boot=args.n_boot))
    boot_within = boot_within[np.isfinite(boot_within)]
    lo_w, hi_w = np.percentile(boot_within, [2.5, 97.5])
    print(f"within-lineage r_partial 95% CI: [{lo_w:.4f}, {hi_w:.4f}]")
    if lo_w > 0 or hi_w < 0:
        print("--> excludes zero: the flux-conditioned I-E dependence survives within a single "
              "organism's own lifetime, not just across the population -- NOT a pure "
              "between-lineage mixture artifact.")
    else:
        print("--> includes zero: report this. The within-lineage estimate is not distinguishable "
              "from zero, meaning the pooled result could be substantially a between-lineage "
              "mixture effect rather than genuine within-organism coupling.")

    print("\n--- (2) within/between covariance decomposition (of flux-residualized I, E) ---")
    decomp = within_between_covariance_decomposition(lid, I, E, fu, fv)
    print(f"total covariance(Delta_I, Delta_E)   = {decomp['total_cov']:.6e}")
    print(f"within-lineage covariance            = {decomp['within_cov']:.6e}  "
          f"({decomp['within_share']*100:.1f}% of total)")
    print(f"between-lineage covariance           = {decomp['between_cov']:.6e}  "
          f"({decomp['between_share']*100:.1f}% of total)")
    print(f"(check: within + between = {decomp['check']:.6e}, should equal total above)")

    print(f"\nrunning lineage-cluster bootstrap ({args.n_boot} resamples) for the within-share...")
    def share_fn(new_lid, I_, E_, fu_, fv_):
        d = within_between_covariance_decomposition(new_lid, I_, E_, fu_, fv_)
        return d["within_share"]
    boot_share = np.array(bootstrap_by_lineage(share_fn, lid, I, E, fu, fv, n_boot=args.n_boot))
    boot_share = boot_share[np.isfinite(boot_share)]
    lo_s, hi_s = np.percentile(boot_share, [2.5, 97.5])
    print(f"within-lineage share of total (flux-residualized) covariance 95% CI: "
          f"[{lo_s*100:.1f}%, {hi_s*100:.1f}%]")

    np.savez("lineage_mixture_decomposition_result.npz",
              r_naive=r_naive, r_within=r_within, n_used=n_used, n_lineages_used=n_lin_used,
              boot_within=boot_within, lo_w=lo_w, hi_w=hi_w,
              within_cov=decomp["within_cov"], between_cov=decomp["between_cov"],
              total_cov=decomp["total_cov"], within_share=decomp["within_share"],
              between_share=decomp["between_share"], boot_share=boot_share, lo_s=lo_s, hi_s=hi_s)
    print("\nsaved lineage_mixture_decomposition_result.npz")


if __name__ == "__main__":
    main()
