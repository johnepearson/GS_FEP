"""
Referee-requested check (both ChatGPT technical reviews, independently
and repeatedly): Table~\\ref{tab:gap_split} in the paper (Section
sec:mechanism_scope) reports the flux-conditioned partial correlation
separately for touching organisms (r=-0.029, screened 99.7%) and
isolated organisms (r=-0.202, screened 44.3%), and the text leans on
the CONTRAST between these two numbers -- "touching organisms are
screened almost completely... isolated organisms are not" -- but the
paper has never put a confidence interval on that contrast itself,
only on each regime's own partial r separately (Table~\\ref{tab:gap_split}
and its surrounding text). This script does that directly, with a
bootstrap that respects the two regimes' actual dependence structure:
touching and isolated organism-instances are NOT independent samples
-- they are two subsets of the SAME 400 sampled frames from the SAME
single trajectory, so a frame that happens to be unusual (e.g. an
especially crowded or especially sparse moment) shifts both subsets'
statistics together. Resampling frames independently for each regime
(as in a two-sample bootstrap of unrelated groups) would overstate how
independent the two point estimates are; this script instead draws ONE
set of resampled frames per trial and uses it for BOTH regimes,
preserving whatever shared frame-level structure exists between them.

Regime split and statistic definitions match the paper's own
gap-split convention exactly (Section sec:mechanism_scope and its
footnote): isolated = positive edge-to-edge gap to nearest neighbor,
touching = zero or negative gap (touching/overlapping); partial
correlation conditions on a single combined flux regressor (PC1 of
standardized Phi_u, Phi_v: ~99% of their joint variance), NOT the two
raw flux variables jointly, because the joint regression is unstable
on gap-stratified subsamples specifically (partial_corr_utils.py's
docstring; the paper's own footnote in sec:mechanism_scope). Screened
fraction is 1-(r_partial/r_marginal)^2, the same convention used
throughout the paper.

For each bootstrap trial, reports:
  - r_partial(isolated), r_partial(touching), and their DIFFERENCE
  - screened_fraction(isolated), screened_fraction(touching), and
    their DIFFERENCE
with percentile 95% CIs on each difference from the shared-frame
bootstrap.

Usage:
    python run_regime_difference_bootstrap.py --data positive_gap_progress_v2.npz
    python run_regime_difference_bootstrap.py --data positive_gap_progress_v2.npz --n-boot 5000
"""
import argparse
import numpy as np

from partial_corr_utils import pc1_partial_corr


def load_gap_dataset(path):
    d = np.load(path)
    I, E, fu, fv = d["I"], d["E"], d["fu"], d["fv"]
    edge_gap, frame_id = d["edge_gap"], d["frame_id"]
    good = (np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
            & np.isfinite(edge_gap))
    return I[good], E[good], fu[good], fv[good], edge_gap[good], frame_id[good]


def regime_stats(I, E, fu, fv):
    """Returns (r_partial, r_marginal, screened_fraction) for one regime's data."""
    if len(I) < 10:
        return np.nan, np.nan, np.nan
    r_partial, _, _ = pc1_partial_corr(I, E, fu, fv)
    r_marginal = np.corrcoef(I, E)[0, 1]
    screened = 1.0 - (r_partial / r_marginal) ** 2 if r_marginal != 0 else np.nan
    return r_partial, r_marginal, screened


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="positive_gap_progress_v2.npz")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    I, E, fu, fv, edge_gap, frame_id = load_gap_dataset(args.data)
    isolated_mask = edge_gap > 0
    touching_mask = ~isolated_mask

    n_iso, n_touch = isolated_mask.sum(), touching_mask.sum()
    print(f"loaded N={len(I)} organism-instances "
          f"({n_iso} isolated [{n_iso/len(I):.1%}], {n_touch} touching/overlapping [{n_touch/len(I):.1%}])")

    r_iso, rm_iso, scr_iso = regime_stats(I[isolated_mask], E[isolated_mask],
                                           fu[isolated_mask], fv[isolated_mask])
    r_touch, rm_touch, scr_touch = regime_stats(I[touching_mask], E[touching_mask],
                                                 fu[touching_mask], fv[touching_mask])

    print(f"\nisolated:  r_partial={r_iso:.4f}  r_marginal={rm_iso:.4f}  screened={scr_iso:.1%}")
    print(f"touching:  r_partial={r_touch:.4f}  r_marginal={rm_touch:.4f}  screened={scr_touch:.1%}")
    print(f"\nobserved difference (touching minus isolated):")
    print(f"  Delta r_partial          = {r_touch - r_iso:+.4f}")
    print(f"  Delta screened fraction  = {(scr_touch - scr_iso)*100:+.1f} percentage points")

    # Shared-frame bootstrap: one resampled set of frames per trial, used
    # for both regimes, so within-frame dependence shared between the two
    # subsets is preserved rather than broken by independent resampling.
    unique_frames = np.unique(frame_id)
    n_frames = len(unique_frames)
    rows_by_frame = {f: np.where(frame_id == f)[0] for f in unique_frames}

    rng = np.random.default_rng(args.seed)
    boot_r_iso = np.empty(args.n_boot)
    boot_r_touch = np.empty(args.n_boot)
    boot_scr_iso = np.empty(args.n_boot)
    boot_scr_touch = np.empty(args.n_boot)

    print(f"\nrunning shared-frame bootstrap ({args.n_boot} resamples, {n_frames} frames)...")
    for b in range(args.n_boot):
        chosen_frames = rng.choice(unique_frames, n_frames, replace=True)
        idx = np.concatenate([rows_by_frame[f] for f in chosen_frames])

        iso_idx = idx[isolated_mask[idx]]
        touch_idx = idx[touching_mask[idx]]

        r_i, _, s_i = regime_stats(I[iso_idx], E[iso_idx], fu[iso_idx], fv[iso_idx])
        r_t, _, s_t = regime_stats(I[touch_idx], E[touch_idx], fu[touch_idx], fv[touch_idx])

        boot_r_iso[b], boot_r_touch[b] = r_i, r_t
        boot_scr_iso[b], boot_scr_touch[b] = s_i, s_t

        if (b + 1) % 500 == 0:
            print(f"  trial {b+1}/{args.n_boot}")

    diff_r = boot_r_touch - boot_r_iso
    diff_scr = boot_scr_touch - boot_scr_iso

    lo_r, hi_r = np.nanpercentile(diff_r, [2.5, 97.5])
    lo_scr, hi_scr = np.nanpercentile(diff_scr, [2.5, 97.5])

    lo_ri, hi_ri = np.nanpercentile(boot_r_iso, [2.5, 97.5])
    lo_rt, hi_rt = np.nanpercentile(boot_r_touch, [2.5, 97.5])

    print(f"\n=== shared-frame bootstrap results ===")
    print(f"isolated r_partial  95% CI: [{lo_ri:.4f}, {hi_ri:.4f}]")
    print(f"touching r_partial  95% CI: [{lo_rt:.4f}, {hi_rt:.4f}]")
    print(f"\nDelta r_partial (touching - isolated)         95% CI: [{lo_r:.4f}, {hi_r:.4f}]")
    print(f"Delta screened fraction (touching - isolated)  95% CI: [{lo_scr*100:.1f}, {hi_scr*100:.1f}] pct points")

    if lo_r > 0 or hi_r < 0:
        print("--> CI on Delta r_partial excludes zero: the regime difference in raw partial "
              "correlation is statistically robust to frame-level clustering.")
    else:
        print("--> CI on Delta r_partial includes zero: report this, the regime difference in "
              "raw partial correlation is NOT statistically distinguishable from zero once "
              "frame-level clustering is accounted for.")

    if lo_scr > 0 or hi_scr < 0:
        print("--> CI on Delta screened-fraction excludes zero: the regime difference in how "
              "completely flux screens the two groups is statistically robust to frame-level "
              "clustering.")
    else:
        print("--> CI on Delta screened-fraction includes zero: report this, the regime "
              "difference in screened fraction is NOT statistically distinguishable from zero "
              "once frame-level clustering is accounted for.")

    np.savez("regime_difference_bootstrap_result.npz",
              r_isolated=r_iso, r_touching=r_touch,
              screened_isolated=scr_iso, screened_touching=scr_touch,
              boot_r_iso=boot_r_iso, boot_r_touch=boot_r_touch,
              boot_scr_iso=boot_scr_iso, boot_scr_touch=boot_scr_touch,
              n_frames=n_frames, n_isolated=n_iso, n_touching=n_touch)
    print("\nsaved regime_difference_bootstrap_result.npz")


if __name__ == "__main__":
    main()
