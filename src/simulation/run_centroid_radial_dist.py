"""
Distributions of (a) centroid speed and (b) radial rate of change dR/dt of
organisms, from the same frame-to-frame matching used for the timescale
separation check (run_remaining_checks.py): organisms (production segmentation: watershed + overlap merge) matched by maximum
overlap (>50% and unambiguous), sampled every SAMPLE_EVERY_STEPS steps.

Per matched transition it records
  speed      centroid displacement per unit time
  dRdt       (R_curr - R_prev)/dt, R = sqrt(area/pi) (equivalent radius)
  R_prev     the previous-frame radius
and writes the arrays plus a two-panel histogram figure.

Usage:
    python run_centroid_radial_dist.py --n-frames 500
    python run_centroid_radial_dist.py --n-frames 40 --out test   # quick test
"""
import argparse
import time
import numpy as np
from params import load as _load_params
_PARAMS = _load_params("initialize.tex")   # F, k, dt of the model

from gray_scott_spectral import PseudoSpectralGS
from gray_scott import GrayScottSim, match_organisms_by_overlap

MIN_PEAK_DISTANCE = 32
PAD = 80
SAMPLE_EVERY_STEPS = 10
CHECKPOINT_FILE = "checkpoint_1024_settled.npz"

import run_ensemble_merged as prod


def segment(proxy):
    """Production segmentation: raw watershed + overlap merge (same as the paper's primary method)."""
    raw, raw_n = proxy.segment_organisms(pad=PAD, min_peak_distance=MIN_PEAK_DISTANCE)
    if raw_n == 0:
        return raw, 0
    labels, n, _ = prod.merge_overlapping_organisms(raw, raw_n, proxy)
    return labels, n


def geometry(proxy, labels, n):
    """Periodic-correct centroids (pixels) and equivalent radii (physical units)."""
    _, radii = proxy.organism_geometry(labels, n)
    return prod.periodic_centroids(labels, n, proxy.n), radii



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--out", default="centroid_radial_dist")
    args = ap.parse_args()

    F, k = _PARAMS["Fval"], _PARAMS["kval"]
    tau = 1.0 / (F + k)

    sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
    proxy.u, proxy.v = sim.u, sim.v
    prev_labels, prev_n = segment(proxy)
    prev_c, prev_r = geometry(proxy, prev_labels, prev_n)
    N = sim.n
    dt_frame = SAMPLE_EVERY_STEPS * sim.dt

    speed, dRdt, Rprev = [], [], []
    t0 = time.time()
    for fr in range(1, args.n_frames + 1):
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        labels, n = segment(proxy)
        if n == 0:
            break
        cen, rad = geometry(proxy, labels, n)
        fwd, bwd = match_organisms_by_overlap(prev_labels, labels, prev_n, n)
        for p, (c, frac) in fwd.items():
            if c is None or frac <= 0.5 or len(bwd.get(c, [])) != 1:
                continue
            d = cen[c - 1] - prev_c[p - 1]
            d = (d + N / 2) % N - N / 2
            r0 = prev_r[p - 1] / proxy.dx
            r1 = rad[c - 1] / proxy.dx
            if r0 > 0:
                speed.append(np.hypot(*d) / dt_frame)
                dRdt.append((r1 - r0) / dt_frame)
                Rprev.append(r0)
        prev_labels, prev_n, prev_c, prev_r = labels, n, cen, rad
        if fr % 20 == 0:
            print(f"frame {fr}/{args.n_frames}  organisms={n}  transitions={len(speed)}  "
                  f"{(time.time()-t0)/fr:.2f}s/frame", flush=True)

    speed, dRdt, Rprev = map(np.array, (speed, dRdt, Rprev))
    np.savez(args.out + ".npz", speed=speed, dRdt=dRdt, R_prev=Rprev, tau=tau)
    rel = dRdt * tau / Rprev
    print(f"\nN={len(speed)} transitions (dt per frame={dt_frame})")
    for name, x in (("centroid speed", speed), ("dR/dt", dRdt), ("|dR/dt|", np.abs(dRdt)),
                    ("rho_centroid = speed*tau/R", speed * tau / Rprev),
                    ("rho_radial = |dR/dt|*tau/R", np.abs(rel))):
        q = np.percentile(x, [1, 10, 50, 90, 99])
        print(f"  {name:28s} mean={x.mean():.4g}  pct[1,10,50,90,99]={np.round(q,4)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    smax = np.percentile(speed, 99.5)
    ax[0].hist(speed, bins=np.linspace(0, smax, 81), color="#4477aa")
    ax[0].set_xlabel("centroid speed"); ax[0].set_ylabel("organism-transitions")
    ax[0].axvline(np.median(speed), color="k", ls="--", lw=1)
    lim = np.percentile(np.abs(dRdt), 99.5)
    ax[1].hist(dRdt, bins=np.linspace(-lim, lim, 81), color="#cc6677")
    ax[1].set_xlabel("radial rate of change dR/dt"); ax[1].axvline(0, color="k", ls=":", lw=1)
    ax[1].axvline(np.median(dRdt), color="k", ls="--", lw=1)
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=200)
    print("saved", args.out + ".npz and .png")


if __name__ == "__main__":
    main()
