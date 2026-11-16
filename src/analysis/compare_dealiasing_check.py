"""
Compares the headline flux-conditioned I-E partial correlation between
the two arms of the dealiasing check (run_dealiasing_check.py --dealias
0 and --dealias 1), both run from the same settled 1024x1024 checkpoint
with the same production sampling. Reports, for each arm: N, marginal
r, joint two-flux partial r (the paper's own headline convention, via
partial_corr_utils.check_partial_corr_stability), and a frame-cluster
bootstrap 95% CI (via cluster_bootstrap_analysis.cluster_bootstrap,
already used elsewhere in this project for the same purpose). Then
reports a frame-cluster bootstrap CI for the DEALIAS-MINUS-NODEALIAS
difference in the partial correlation, by drawing one bootstrap
replicate independently from each arm's own frame clusters per trial
and differencing -- valid because the two arms are independent runs
(different random trajectories after the first few steps), not paired
observations.

The two trajectories are NOT expected to track each other pointwise
(chaotic system); what this script checks is whether the DISTRIBUTIONAL
statistic -- the thing the paper actually reports -- moves by more than
its own sampling noise once dealiasing is turned on.

Usage:
    python compare_dealiasing_check.py \\
        --nodealias dealiasing_check_nodealias_progress.npz \\
        --dealias dealiasing_check_dealias_progress.npz
"""
import argparse
import numpy as np

from partial_corr_utils import check_partial_corr_stability
from cluster_bootstrap_analysis import cluster_bootstrap


def load_clean(path):
    d = np.load(path)
    I, E, fu, fv, frame_id = d["I"], d["E"], d["fu"], d["fv"], d["frame_id"]
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    return I[good], E[good], fu[good], fv[good], frame_id[good], int(d["frames_done"])


def summarize(label, I, E, fu, fv, frame_id, n_boot=2000, seed=0):
    r_both, r_fu, r_fv, cond, is_suspect = check_partial_corr_stability(I, E, fu, fv)
    r_marginal = np.corrcoef(I, E)[0, 1]
    screened = 1.0 - (r_both / r_marginal) ** 2 if r_marginal != 0 else float("nan")
    boot_r, n_clusters = cluster_bootstrap(I, E, fu, fv, frame_id, n_boot=n_boot, seed=seed)
    lo, hi = np.percentile(boot_r, [2.5, 97.5])
    print(f"\n=== {label} ===")
    print(f"  N = {len(I)}  (frames = {n_clusters})")
    print(f"  marginal r          = {r_marginal:.4f}")
    print(f"  joint partial r     = {r_both:.4f}  "
          f"(single-variable range: [{min(r_fu, r_fv):.4f}, {max(r_fu, r_fv):.4f}]"
          f"{', OUTSIDE this range -- flag as for the isolated-subset collider check' if is_suspect else ', within range'})")
    print(f"  screened fraction   = {screened:.1%}")
    print(f"  frame-cluster bootstrap 95% CI on partial r: [{lo:.4f}, {hi:.4f}]")
    return dict(r_both=r_both, r_marginal=r_marginal, screened=screened,
                boot_r=boot_r, n=len(I), n_clusters=n_clusters,
                cond=cond, is_suspect=is_suspect)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodealias", required=True)
    ap.add_argument("--dealias", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    I0, E0, fu0, fv0, fr0, nf0 = load_clean(args.nodealias)
    I1, E1, fu1, fv1, fr1, nf1 = load_clean(args.dealias)

    print(f"nodealias arm: {nf0} frames requested, {len(np.unique(fr0))} with >=1 organism")
    print(f"dealias arm:   {nf1} frames requested, {len(np.unique(fr1))} with >=1 organism")

    res0 = summarize("no dealiasing (existing solver behavior)", I0, E0, fu0, fv0, fr0, n_boot=args.n_boot)
    res1 = summarize("2/3-rule dealiasing enabled", I1, E1, fu1, fv1, fr1, n_boot=args.n_boot)

    # Difference in the headline statistic, with its own bootstrap CI:
    # draw one bootstrap replicate independently from EACH arm's own
    # frame clusters per trial (valid since the two arms are
    # independent runs), and difference them.
    rng = np.random.default_rng(0)
    n_boot = min(len(res0["boot_r"]), len(res1["boot_r"]))
    diff = res1["boot_r"][:n_boot] - res0["boot_r"][:n_boot]
    lo, hi = np.percentile(diff, [2.5, 97.5])
    print("\n=== dealias minus no-dealias, joint partial r ===")
    print(f"  point estimate      = {res1['r_both'] - res0['r_both']:.4f}")
    print(f"  bootstrap 95% CI    = [{lo:.4f}, {hi:.4f}]")
    if lo - 1e-10 <= 0.0 <= hi + 1e-10:
        print("  --> CI includes zero: no evidence dealiasing materially changes the headline statistic.")
    else:
        print("  --> CI excludes zero: dealiasing DOES materially change the headline statistic -- "
              "report this, do not average over it or discard either run.")

    from texout import num, write_macros
    write_macros("results/dealiascmp.tex", "compare_dealiasing_check.py", [
        ("DealiasR", num(res1["r_both"], 3)), ("DealiasN", num(res1["n"], 0)),
        ("DealiasPlainR", num(res0["r_both"], 3)), ("DealiasPlainN", num(res0["n"], 0)),
        ("DealiasDiff", num(res1["r_both"] - res0["r_both"], 3)),
        ("DealiasDiffLo", num(lo, 3)), ("DealiasDiffHi", num(hi, 3)),
        ("DealiasLo", num(np.percentile(res1["boot_r"], 2.5), 3)), ("DealiasHi", num(np.percentile(res1["boot_r"], 97.5), 3)),
        ("DealiasPlainLo", num(np.percentile(res0["boot_r"], 2.5), 3)), ("DealiasPlainHi", num(np.percentile(res0["boot_r"], 97.5), 3))])
    print("\n(Raw numbers above are what belong in the paper; do not round further before recording them.)")


if __name__ == "__main__":
    main()
