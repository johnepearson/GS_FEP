"""
Screening curve (paper: Figure of the richer-blanket test, ESM section "The Richer-Blanket Screening Curve").

Partial correlation between an organism's interior state I and its far-field exterior (mean v over the nine shells
at 9.2-18.2 organism radii), conditioning on the PC1 flux regressor plus the organism's own mean v in the first n
near-field shells (8-px shells out to 128 px, i.e. 5.8 organism radii with R = 21.3 px), for n = 0 (flux alone) to 16.
Same linear partial-correlation construction as everywhere else: regress I and the far-field exterior on
[1, PC1, shell_1 ... shell_n] and correlate the residuals.

Reads master_trajectory_progress.npz (written by run_master_trajectory.py).  Prints the values so they can be
checked against the supplement (flux alone -0.050; +0.056, +0.089, +0.114 for n = 1, 2, 3; about +0.10 after).

Usage:  python3 plot_screening_curve.py
        python3 plot_screening_curve.py --data master_trajectory_progress.npz --out screening_curve.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R_PX = 21.3
N_NEAR = 16
FAR_LO_PX, FAR_HI_PX = 190.0, 390.0     # mids of the nine far-field shells are 196 ... 388 px


def residualize(y, B):
    c, *_ = np.linalg.lstsq(B, y, rcond=None)
    return y - B @ c


def pc1_flux(fu, fv):
    Z = np.column_stack([(fu - fu.mean()) / fu.std(), (fv - fv.mean()) / fv.std()])
    w, V = np.linalg.eigh(np.cov(Z.T))
    return Z @ V[:, np.argmax(w)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="master_trajectory_progress.npz")
    ap.add_argument("--out", default="screening_curve.png")
    a = ap.parse_args()
    d = np.load(a.data)
    I, fu, fv, vs, edges = d["I"], d["fu"], d["fv"], d["v_shells"], d["radii"].astype(float)
    mids = 0.5 * (edges[:-1] + edges[1:])
    far_idx = np.where((mids >= FAR_LO_PX) & (mids <= FAR_HI_PX))[0]
    assert len(far_idx) == 9, f"expected nine far-field shells, found {len(far_idx)}"
    E_far = vs[:, far_idx].mean(1)
    near = vs[:, :N_NEAR]
    good = np.isfinite(I) & np.isfinite(fu) & np.isfinite(fv) & np.isfinite(E_far) & np.isfinite(near).all(1)
    I, fu, fv, E_far, near = I[good], fu[good], fv[good], E_far[good], near[good]
    N = len(I)
    pc1 = pc1_flux(fu, fv)

    ns = np.arange(0, N_NEAR + 1)
    r = []
    for n in ns:
        B = np.column_stack([np.ones(N), pc1] + [near[:, j] for j in range(n)])
        r.append(np.corrcoef(residualize(I, B), residualize(E_far, B))[0, 1])
    r = np.array(r)
    x = np.concatenate([[0.0], mids[:N_NEAR] / R_PX])

    print(f"N = {N:,}; far-field shells at {mids[far_idx][0]:.0f}-{mids[far_idx][-1]:.0f} px; R = {R_PX} px")
    for n, xi, ri in zip(ns, x, r):
        print(f"  n = {n:2d}  outermost shell {xi:5.2f} radii   r = {ri:+.4f}")

    fig, ax = plt.subplots(figsize=(6.4, 3.85))
    ax.axhline(0, color="0.55", lw=0.8, zorder=1)
    ax.plot(x, r, "o-", color="#1f4e8c", lw=1.6, ms=3.6, zorder=2)
    ax.annotate("flux alone", xy=(x[0], r[0]), xytext=(0.4, r[0] + 0.01), fontsize=7,
                arrowprops=dict(arrowstyle="-", color="0.5", lw=1))
    ax.set_xlabel("Outermost near-field shell included (organism radii from the boundary)")
    ax.set_ylabel("Partial correlation of $I$ with far-field exterior")
    ax.set_title(f"Screening curve (N = {N:,})", fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print("saved", a.out)


if __name__ == "__main__":
    main()
