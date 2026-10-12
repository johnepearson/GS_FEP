"""
Recomputes main text Figure 7 (fig:decay_curve, \\S3.4, "Spatial Structure
of the Residual Coupling") using the PC1 flux regressor instead of the
joint two-variable (Phi_u,Phi_v) regression that plot_decay_curve_merged.py
used, for consistency with the rest of the revision's PC1 convention
(\\S3.3.1's footnote; check_joint_regression_stability.py showed the joint
regression is unstable for the single-trajectory and both ensemble
headline datasets, and ESM \\S2.2 already flagged it for the isolated
gap-split subset).

No new simulation: reuses the already-collected
decay_curve_merged_progress.npz (same file run_decay_curve_merged.py /
plot_decay_curve_merged.py already used), just changes the regressor
per shell.

For EACH shell, this also reports the joint-regression estimate and
whether it falls inside the single-variable [Phi_u-alone, Phi_v-alone]
range (the same stability check used everywhere else this revision), so
we know whether the joint-vs-PC1 difference at any given shell is the
collinearity artifact or just normal estimation noise.

Produces decay_curve_1024.png (same filename as before -- overwrite in
place) and prints a full shell-by-shell report.

Usage:
    python plot_decay_curve_pc1.py --data decay_curve_merged_progress.npz
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORGANISM_RADIUS_PX = 20.0  # median organism radius at 1024x1024, used project-wide


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
    ap.add_argument("--data", default="decay_curve_merged_progress.npz")
    ap.add_argument("--out", default="decay_curve_1024.png")
    args = ap.parse_args()

    d = np.load(args.data)
    I_all, fu_all, fv_all, profiles_all, radii_edges = (
        d["I"], d["fu"], d["fv"], d["profiles"], d["radii"])
    good = np.isfinite(I_all) & np.isfinite(fu_all) & np.isfinite(fv_all)
    I, fu, fv = I_all[good], fu_all[good], fv_all[good]
    N_total = len(I)
    print(f"N={N_total} organism-instances")

    n_shells = profiles_all.shape[1]
    mids_px, rs, los, his, ns, n_unstable = [], [], [], [], [], 0
    for s in range(n_shells):
        shellV = profiles_all[good, s]
        g = np.isfinite(shellV)
        n = g.sum()
        if n < 30:
            continue
        Ig, Eg, fug, fvg = I[g], shellV[g], fu[g], fv[g]

        r_u_alone = partial_r_single(Ig, Eg, fug)
        r_v_alone = partial_r_single(Ig, Eg, fvg)
        r_joint = partial_r_joint(Ig, Eg, fug, fvg)
        lo_range, hi_range = min(r_u_alone, r_v_alone), max(r_u_alone, r_v_alone)
        joint_stable = lo_range <= r_joint <= hi_range
        if not joint_stable:
            n_unstable += 1

        pc1, var_explained = pc1_of(fug, fvg)
        r = partial_r_single(Ig, Eg, pc1)
        lo, hi = fisher_ci(r, n)

        mid = 0.5 * (radii_edges[s] + radii_edges[s + 1])
        mids_px.append(mid); rs.append(r); los.append(lo); his.append(hi); ns.append(n)
        print(f"  shell mid={mid:6.0f}px ({mid/ORGANISM_RADIUS_PX:5.2f} radii)  "
              f"PC1 r={r:+.4f}  CI=[{lo:+.4f},{hi:+.4f}]  n={n}  "
              f"[joint r={r_joint:+.4f} {'OK' if joint_stable else 'UNSTABLE'}]")

    print(f"\n{n_unstable}/{len(mids_px)} shells had an unstable joint-regression estimate "
          f"by the [min(r_u),max(r_v)] criterion.")

    mids_px = np.array(mids_px)
    rs = np.array(rs)
    los = np.array(los)
    his = np.array(his)
    mids_radii = mids_px / ORGANISM_RADIUS_PX

    print(f"\nnear-boundary region (<=2 radii): min r = {rs[mids_radii<=2].min():+.4f} "
          f"at {mids_radii[mids_radii<=2][np.argmin(rs[mids_radii<=2])]:.2f} radii")
    print(f"innermost shell: r={rs[0]:+.4f} at {mids_radii[0]:.2f} radii")
    print(f"overall min: r={rs.min():+.4f} at {mids_radii[np.argmin(rs)]:.2f} radii")
    print(f"overall max: r={rs.max():+.4f} at {mids_radii[np.argmax(rs)]:.2f} radii")

    np.savez("decay_curve_pc1_results.npz", mids_radii=mids_radii, r=rs, lo=los, hi=his, n=np.array(ns))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)
    ax.errorbar(mids_radii, rs, yerr=[rs - los, his - rs], fmt="o-", color="crimson",
                ms=4, lw=1.3, capsize=2)
    ax.set_xlabel("distance from boundary (organism radii)")
    ax.set_ylabel("partial correlation $r(s)$")
    ax.set_title(f"Interior/exterior partial correlation vs. distance from boundary\n"
                 f"merge-based segmentation, PC1 flux regressor, $N={N_total:,}$ organism-instances")
    plt.tight_layout()
    plt.savefig(args.out, dpi=140)
    plt.close(fig)
    print(f"\nsaved {args.out}")
    print(f"saved decay_curve_pc1_results.npz")


if __name__ == "__main__":
    main()
