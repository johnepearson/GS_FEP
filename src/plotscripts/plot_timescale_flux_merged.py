"""
Regenerates the two Section 3.1/3.2 figures (fig:timescale_separation,
fig:flux_asymmetry) under the merge-based segmentation, from
run_remaining_checks_merged.py's saved output
(remaining_checks_merged_results.npz: rho, flux_u_neg, flux_v_pos,
boundary_v, edge_gap, merge_size). Matches the style of the
pre-merge-fix figures exactly (same titles, axis labels, layout) so
the only visible change is the data and the annotated median.

Produces:
  timescale_separation_1024.png -- histogram of rho (N=92,125),
      vertical dashed line at the median (0.032, was 0.035).
  flux_asymmetry_1024.png -- per-frame fraction of organisms satisfying
      the U-inward / V-outward flux asymmetry, 500 frames.

Usage:
    python plot_timescale_flux_merged.py --data remaining_checks_merged_results.npz
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="remaining_checks_merged_results.npz")
    args = ap.parse_args()

    d = np.load(args.data)
    rho = d["rho"]
    flux_u_neg = d["flux_u_neg"]
    flux_v_pos = d["flux_v_pos"]

    median_rho = np.median(rho)
    p90, p99 = np.percentile(rho, [90, 99])
    print(f"Timescale separation: N={len(rho)}  median rho={median_rho:.4f}  "
          f"90th pct={p90:.4f}  99th pct={p99:.4f}")
    print(f"Flux asymmetry: mean fraction U-inward={flux_u_neg.mean():.4f}  "
          f"mean fraction V-outward={flux_v_pos.mean():.4f}")

    # --- figure 1: timescale separation histogram ---
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.hist(rho, bins=40, range=(0, 0.6), color="steelblue", edgecolor="white")
    ax.axvline(median_rho, color="crimson", ls="--", lw=1.6,
               label=f"median={median_rho:.3f}")
    ax.set_xlabel(r"$\rho$ = boundary displacement per relaxation time / radius")
    ax.set_ylabel("count")
    ax.set_title("Timescale separation")
    ax.legend()
    plt.tight_layout()
    plt.savefig("timescale_separation_1024.png", dpi=140)
    plt.close(fig)

    # --- figure 2: flux asymmetry per frame ---
    frames = np.arange(len(flux_u_neg))
    fig2, ax2 = plt.subplots(figsize=(6.5, 5.2))
    ax2.plot(frames, flux_u_neg, color="tab:blue", lw=1, label="fraction U-inward")
    ax2.plot(frames, flux_v_pos, color="tab:orange", lw=1, label="fraction V-outward")
    ax2.set_xlabel("frame")
    ax2.set_ylabel("fraction of organisms")
    ax2.set_title("Flux asymmetry")
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("flux_asymmetry_1024.png", dpi=140)
    plt.close(fig2)

    print("\nsaved timescale_separation_1024.png and flux_asymmetry_1024.png")


if __name__ == "__main__":
    main()
