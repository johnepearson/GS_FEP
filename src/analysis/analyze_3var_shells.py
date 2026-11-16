"""
Correlation-length comparison for the three-variable chemical-memory model, at the current individuation
(marker spacing 24, merged organisms), against the plain Gray-Scott baseline from the master trajectory.

r(s) = partial correlation (PC1 flux regressor) of the organism's interior state I with the mean v in the
concentric shell at distance s, exactly as in analyze_master.py section 5.

Inputs (all in local_package/):
    master_trajectory_progress.npz         plain Gray-Scott, 1024^2, spacing 24, merged   (baseline)
    shell_data_3var_1024_Dw0_md24.npz      memory model, D_w = 0
    shell_data_3var_1024_DwDv_md24.npz     memory model, D_w = D_v
Output: memory_comparison_1024.png and a printed table.

Usage:  python3 analyze_3var_shells.py
"""
import numpy as np
from partial_corr_utils import pc1_partial_corr


NCOUNT = []


def curve(I, fu, fv, vs, radii, R, label):
    good = np.isfinite(I) & np.isfinite(fu) & np.isfinite(fv)
    xs, rs, ns = [], [], []
    for s in range(vs.shape[1]):
        m = good & np.isfinite(vs[:, s])
        if m.sum() > 100:
            rs.append(pc1_partial_corr(I[m], vs[m, s], fu[m], fv[m])[0])
            xs.append(0.5 * (radii[s] + radii[s + 1]) / R); ns.append(m.sum())
    xs, rs = np.array(xs), np.array(rs)
    NCOUNT.append(int(good.sum()))
    print(f"\n{label}: N={good.sum():,} organism-instances; median organism radius {R:.1f} px")
    i = np.argmin(rs)
    print(f"  deepest value r={rs[i]:+.3f} at {xs[i]:.1f} radii")
    for target in (1.0, 1.4, 1.8, 3.0, 5.0, 7.4, 10.0, 13.4, 19.4):
        j = np.argmin(np.abs(xs - target))
        print(f"  r at {xs[j]:5.2f} radii = {rs[j]:+.3f}")
    far = xs >= 7
    print(f"  7+ radii: r between {rs[far].min():+.3f} and {rs[far].max():+.3f}")
    return xs, rs


def main():
    out = {}
    d = np.load("master_trajectory_progress.npz")
    out["plain Gray-Scott"] = curve(d["I"], d["fu"], d["fv"], d["v_shells"], d["radii"],
                                    np.nanmedian(d["radius_px"]), "plain Gray-Scott (master trajectory)")
    for tag, name in (("Dw0", "memory, $D_w=0$"), ("DwDv", "memory, $D_w=D_v$")):
        d = np.load(f"shell_data_3var_1024_{tag}_md24.npz")
        out[name] = curve(d["I"], d["fu"], d["fv"], d["profiles"], d["radii"],
                          np.nanmedian(d["radius_px"]), f"memory model {tag}")
    CV = {}
    for tag in ("Dw0", "DwDv"):
        d = np.load(f"shell_data_3var_1024_{tag}_md24.npz")
        r = d["radius_px"]
        CV[tag] = (np.nanstd(r) / np.nanmean(r), np.nanmean(r))
        print(f"{tag}: size CV (SD/mean of organism radius) = {np.nanstd(r)/np.nanmean(r):.3f}, mean radius {np.nanmean(r):.1f} px")
    from texout import num, write_macros
    write_macros("results/shells3var.tex", "analyze_3var_shells.py",
                 [("ShellInnerMemory", num(out["memory, $D_w=D_v$"][1][0], 4)),
                  ("ShellInnerPlain", num(out["plain Gray-Scott"][1][0], 4)),
                  ("ShellCVMemory", num(CV["DwDv"][0], 3)), ("ShellCVZero", num(CV["Dw0"][0], 3)),
                  ("ShellRadiusZero", num(CV["Dw0"][1], 1)),
                  ("ShellInnerRadii", num(out["memory, $D_w=D_v$"][0][0], 1)),
                  ("ShellMemNearOne", num(out["memory, $D_w=D_v$"][1][np.argmin(np.abs(out["memory, $D_w=D_v$"][0] - 1.1))], 3)),
                  ("ShellMemNearOneAt", num(out["memory, $D_w=D_v$"][0][np.argmin(np.abs(out["memory, $D_w=D_v$"][0] - 1.1))], 1)),
                  ("ShellMemNearThree", num(out["memory, $D_w=D_v$"][1][np.argmin(np.abs(out["memory, $D_w=D_v$"][0] - 2.9))], 3)),
                  ("ShellMemNearThreeAt", num(out["memory, $D_w=D_v$"][0][np.argmin(np.abs(out["memory, $D_w=D_v$"][0] - 2.9))], 1)),
                  ("ShellMemFarLo", num(out["memory, $D_w=D_v$"][1][(out["memory, $D_w=D_v$"][0] >= 7) & (out["memory, $D_w=D_v$"][0] <= 15)].min(), 3)),
                  ("ShellMemFarHi", num(out["memory, $D_w=D_v$"][1][(out["memory, $D_w=D_v$"][0] >= 7) & (out["memory, $D_w=D_v$"][0] <= 15)].max(), 3)),
                  ("ShellPlainNearLo", num(out["plain Gray-Scott"][1][(out["plain Gray-Scott"][0] >= 1) & (out["plain Gray-Scott"][0] <= 3)].min(), 2)),
                  ("ShellPlainNearHi", num(out["plain Gray-Scott"][1][(out["plain Gray-Scott"][0] >= 1) & (out["plain Gray-Scott"][0] <= 3)].max(), 2)),
                  ("ShellPlainFarLo", num(out["plain Gray-Scott"][1][out["plain Gray-Scott"][0] >= 7].min(), 2)),
                  ("ShellPlainFarHi", num(out["plain Gray-Scott"][1][out["plain Gray-Scott"][0] >= 7].max(), 2)),
                  ("ShellZeroMin", num(out["memory, $D_w=0$"][1].min(), 2)), ("ShellZeroMax", num(out["memory, $D_w=0$"][1].max(), 2)),
                  ("ShellNPlain", num(NCOUNT[0], 0)), ("ShellNZero", num(NCOUNT[1], 0)), ("ShellNMemory", num(NCOUNT[2], 0))])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    for name, c in (("plain Gray-Scott", "0.35"), ("memory, $D_w=D_v$", "#1f4e8c")):
        x, r = out[name]
        ax[0].plot(x, r, "o-", ms=3, color=c, label=name)
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("distance from the organism's boundary (organism radii)")
    ax[0].set_ylabel("partial correlation $r(s)$ with interior state $I$")
    ax[0].set_title("populations that replicate", fontsize=10)
    ax[0].legend(frameon=False)
    x, r = out["memory, $D_w=0$"]
    ax[1].plot(x, r, "o-", ms=3, color="#c0392b", label="memory, $D_w=0$ (frozen lattice)")
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].set_xlabel("distance from the organism's boundary (organism radii)")
    ax[1].set_title("$D_w=0$: near-regular lattice, no replication", fontsize=10)
    ax[1].legend(frameon=False)
    fig.tight_layout(); fig.savefig("memory_comparison_1024.png", dpi=200)
    print("\nwrote memory_comparison_1024.png")


if __name__ == "__main__":
    main()
