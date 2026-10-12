"""
Figures for the blanket-thickness sweep (ESM Figure S_band) and the substrate-enriched CMI (ESM Figure S_u_enriched).

Reads the outputs of the CMI scripts (no new computation):
    run_band_sweep_seed0.npz, run_band_sweep_seed1.npz, run_band_sweep_seed2.npz     (run_band_sweep_cmi.py --seed 0/1/2)
    u_enriched_cmi_ring.npz                                                        (run_u_enriched_cmi.py)
and writes band_sweep_cmi.png and u_enriched_cmi.png.  In the band figure the points are means over the seeds and the bars on
the real-band curve show the range over the seeds; the dotted vertical line marks one organism radius (21.3 px).

Usage (from local_package):
    python3 plot_blanket_figures.py
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY = "#1f4e79", "#c8552b", "#777777"
WIDTHS = [0, 8, 16, 24, 40, 64, 96]
RADIUS_PX = 21.3


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="k", lw=0.7, zorder=0)


def band_figure(seeds, out):
    data = [np.load(f"run_band_sweep_seed{s}.npz") for s in seeds]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (name, title) in zip(axes, (("all", "all organisms"), ("isolated", "unmerged organisms"))):
        def vals(kind, w):
            k = f"{name}_w{w}_{kind}_obs"
            return np.array([float(d[k]) for d in data if k in d.files])
        real = [vals("real", w) for w in WIDTHS]
        m = np.array([r.mean() for r in real])
        lo = m - np.array([r.min() for r in real])
        hi = np.array([r.max() for r in real]) - m
        ax.errorbar(WIDTHS, m, yerr=[lo, hi], fmt="o-", color=BLUE, lw=2, ms=6, capsize=3,
                    label="real band ($u$ and $v$ summaries)")
        w2 = WIDTHS[1:]
        ax.plot(w2, [vals("noise", w).mean() for w in w2], "^:", color=GREY, lw=2, ms=7,
                label="same dimension, band information removed")
        ax.plot(w2, [vals("ecopy", w).mean() for w in w2], "s--", color=ORANGE, lw=2, ms=6,
                label="noisy copy of $E$ (same $R^2$)")
        ax.axvline(RADIUS_PX, color="k", lw=0.8, ls="-.")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("band width $w$ beyond the boundary (px)")
        clean(ax)
    axes[0].set_ylabel("CMI $I(I;E_w\\mid Z_w)$ (nats)")
    axes[0].text(RADIUS_PX + 1, 0.15, "1 organism\nradius", fontsize=8)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ("real band ($u$ and $v$ summaries)", "same dimension, band information removed",
                                  "noisy copy of $E$ (same $R^2$)")]
    axes[0].legend([h[i] for i in order], [l[i] for i in order], loc="center right", fontsize=8.5, frameon=False,
                   bbox_to_anchor=(1.0, 0.55))
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("saved", out)


def u_figure(path, out):
    d = np.load(path)
    levels = [0, 1, 2, 3, 5, 7]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    for ax, (name, title) in zip(axes, (("all", "all organisms ($N=8{,}000$ subsample of 96,409)"),
                                        ("isolated", "unmerged organisms"))):
        g = lambda kind, n: float(d[f"{name}_{kind}{n}_obs"])
        ax.plot([2 + n for n in levels], [g("u", n) for n in levels], "o-", color=BLUE, lw=2, ms=6, label="real $u$-shells")
        ax.plot([5, 9], [g("ecopy", 3), g("ecopy", 7)], "s--", color=ORANGE, lw=2, ms=6, label="noisy copy of $E$ (same $R^2$)")
        ax.plot([5, 9], [g("noise", 3), g("noise", 7)], "^:", color=GREY, lw=2, ms=7, label="pure-noise columns")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("dimension of conditioning set (2 = flux only)")
        clean(ax)
    axes[0].set_ylabel("CMI $I(I;E\\mid Z)$ (nats)")
    axes[0].legend(loc="upper right", fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--u-file", default="u_enriched_cmi_ring.npz")
    a = ap.parse_args()
    band_figure(a.seeds, "band_sweep_cmi.png")
    u_figure(a.u_file, "u_enriched_cmi.png")
