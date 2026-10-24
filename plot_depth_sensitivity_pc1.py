"""
Draws ESM Figure S1 (fig:S_depth_sensitivity), the depth-fraction robustness check of the interior definition (main
text §2.1): the r(d) curve over core depth-fractions d in {0.5, 0.6, 0.7, 0.8, 0.9}, using the PC1 flux regressor
(main text §3.3.1).

No new simulation: reads sweep_merged_progress.npz (written by sweep_from_master.py).

Also reports, at each depth-fraction, the joint-regression estimate and whether it passes the stability check (falls
within [min(r_Phi_u-alone, r_Phi_v-alone), max(...)]).

Produces depth_sensitivity.png and a full per-fraction report.

Usage:
    python plot_depth_sensitivity_pc1.py --data sweep_merged_progress.npz
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FRACTIONS = [0.50, 0.60, 0.70, 0.80, 0.90]


def residualize(y, B):
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def pc1_of(fu, fv):
    B_std = np.column_stack([
        (fu - fu.mean()) / fu.std(),
        (fv - fv.mean()) / fv.std(),
    ])
    cov = np.cov(B_std.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1_dir = eigvecs[:, np.argmax(eigvals)]
    pc1 = B_std @ pc1_dir
    var_explained = eigvals.max() / eigvals.sum()
    return pc1, var_explained


def partial_r_single(y1, y2, x):
    N = len(y1)
    B = np.column_stack([np.ones(N), x])
    return np.corrcoef(residualize(y1, B), residualize(y2, B))[0, 1]


def partial_r_joint(y1, y2, fu, fv):
    N = len(y1)
    B = np.column_stack([np.ones(N), fu, fv])
    return np.corrcoef(residualize(y1, B), residualize(y2, B))[0, 1]


def fisher_ci(r, n):
    se = 1 / np.sqrt(n - 3)
    z = np.arctanh(r)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="sweep_merged_progress.npz")
    ap.add_argument("--out", default="depth_sensitivity.png")
    args = ap.parse_args()

    d = np.load(args.data)

    ds, rs, los, his, Ns = [], [], [], [], []
    n_unstable = 0
    for f in FRACTIONS:
        I, E, fu, fv = d[f"{f}_I"], d[f"{f}_E"], d[f"{f}_fu"], d[f"{f}_fv"]
        good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
        I, E, fu, fv = I[good], E[good], fu[good], fv[good]
        N = len(I)

        r_u_alone = partial_r_single(I, E, fu)
        r_v_alone = partial_r_single(I, E, fv)
        r_joint = partial_r_joint(I, E, fu, fv)
        lo_range, hi_range = min(r_u_alone, r_v_alone), max(r_u_alone, r_v_alone)
        joint_stable = lo_range <= r_joint <= hi_range
        if not joint_stable:
            n_unstable += 1

        pc1, var_explained = pc1_of(fu, fv)
        r = partial_r_single(I, E, pc1)
        lo, hi = fisher_ci(r, N)

        ds.append(f); rs.append(r); los.append(lo); his.append(hi); Ns.append(N)
        print(f"d={f:.2f}  N={N:>6}  PC1 r={r:.4f}  95% CI=[{lo:.4f},{hi:.4f}]  "
              f"(PC1 explains {100*var_explained:.1f}% of joint var)  "
              f"[joint r={r_joint:+.4f} {'OK' if joint_stable else 'UNSTABLE'}]")

    print(f"\n{n_unstable}/{len(FRACTIONS)} depth-fractions had an unstable joint-regression "
          f"estimate by the [min(r_u),max(r_v)] criterion.")

    ds = np.array(ds); rs = np.array(rs); los = np.array(los); his = np.array(his)
    print(f"\nrange: r={rs.min():.4f} at d={ds[np.argmin(rs)]:.2f} to "
          f"r={rs.max():.4f} at d={ds[np.argmax(rs)]:.2f}")
    print(f"d=0.5 -> d=0.9 change: {rs[0]:.4f} -> {rs[-1]:.4f}")

    np.savez("depth_sensitivity_pc1_results.npz", d=ds, r=rs, lo=los, hi=his, n=np.array(Ns))

    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.errorbar(ds, rs, yerr=[rs - los, his - rs], fmt="o-", color="crimson",
                ms=6, lw=1.6, capsize=3)
    ax.set_xlabel("core depth-fraction $d$")
    ax.set_ylabel("partial correlation $r(d)$")
    ax.set_title(f"Partial correlation vs. core depth\n(merge-based segmentation, PC1 flux regressor,\n"
                 f"$N={Ns[0]:,}$ organism-instances per point)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {args.out}")
    print("saved depth_sensitivity_pc1_results.npz")


if __name__ == "__main__":
    main()
