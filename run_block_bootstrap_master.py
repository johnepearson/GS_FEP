"""
Moving-block bootstrap for the single-trajectory (master) partial correlation (reviewer request).

Frames of one trajectory are serially dependent, so resampling single frames (the paper's frame-cluster bootstrap)
can understate the uncertainty. Here whole BLOCKS of consecutive frames are resampled (circular moving blocks),
with block lengths 1, 5, 10, 20 and 40 frames. The statistic is the I-E partial correlation controlling for the PC1
flux regressor (PC1 direction fixed from the full data; recomputing it per resample changes nothing visible).
Fast: per-frame sufficient statistics (counts, sums, cross-products of I, E, PC1) are summed per resample.

Usage (from local_package/, master_trajectory_progress.npz present):
    python3 run_block_bootstrap_master.py
"""
import numpy as np
from partial_corr_utils import pc1_partial_corr

N_BOOT = 4000
BLOCKS = [1, 5, 10, 20, 40]


def pcorr_from_moments(n, S, Q):
    """n, S (3,), Q (3,3) sums over rows of [I,E,P] -> partial corr of I,E given P (with intercept)."""
    m = S / n
    C = Q / n - np.outer(m, m)
    return (C[0, 1] - C[0, 2] * C[1, 2] / C[2, 2]) / np.sqrt((C[0, 0] - C[0, 2] ** 2 / C[2, 2]) * (C[1, 1] - C[1, 2] ** 2 / C[2, 2]))


def run(label, I, E, fu, fv, frame, rng):
    good = np.isfinite(I) & np.isfinite(E) & np.isfinite(fu) & np.isfinite(fv)
    I, E, fu, fv, frame = I[good], E[good], fu[good], fv[good], frame[good]
    r0, n0, expl, (mean, std, v1) = pc1_partial_corr(I, E, fu, fv, return_direction=True)
    P = ((np.column_stack([fu, fv]) - mean) / std) @ v1
    X = np.column_stack([I, E, P])
    frames = np.unique(frame)
    F = len(frames)
    n = np.zeros(F); S = np.zeros((F, 3)); Q = np.zeros((F, 3, 3))
    for k, f in enumerate(frames):
        x = X[frame == f]
        n[k] = len(x); S[k] = x.sum(0); Q[k] = x.T @ x
    print(f"\n{label}: {n0:,} organism-instances in {F} frames; PC1 partial r = {r0:+.4f}")
    for b in BLOCKS:
        nb = int(np.ceil(F / b))
        out = np.empty(N_BOOT)
        for t in range(N_BOOT):
            starts = rng.integers(0, F, nb)
            idx = ((starts[:, None] + np.arange(b)[None, :]) % F).ravel()[:F]
            out[t] = pcorr_from_moments(n[idx].sum(), S[idx].sum(0), Q[idx].sum(0))
        lo, hi = np.percentile(out, [2.5, 97.5])
        print(f"  block length {b:3d} frames: bootstrap SD {out.std():.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
    z = 1.0 / np.sqrt(n0 - 3)
    print(f"  (naive instance-level SE of r, for reference: {z:.4f})")


def main():
    d = np.load("master_trajectory_progress.npz")
    rng = np.random.default_rng(0)
    I, E, fu, fv, fr, ms = d["I"], d["E"], d["fu"], d["fv"], d["frame_id"], d["merge_size"]
    run("All organisms", I, E, fu, fv, fr, rng)
    m = ms == 1
    run("Isolated organisms (merge_size == 1)", I[m], E[m], fu[m], fv[m], fr[m], rng)


if __name__ == "__main__":
    main()
