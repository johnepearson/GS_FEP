"""
Distribution of the area-averaged rate of change of v over each organism.

Organisms are the production (watershed + overlap-merge) spots.
Same frame matching as run_remaining_checks.py / run_centroid_radial_dist.py
(overlap > 50%, unambiguous, sampled every SAMPLE_EVERY_STEPS steps).
Per matched transition p -> c it records

  dv_fixed   mean over the PREVIOUS mask of (v_now - v_prev)/dt
             (area average of dv/dt on a fixed region)
  dv_comove  (mean v over current mask - mean v over previous mask)/dt
             (rate of change of the organism's own mean v; follows the blob)
  vbar_prev  mean v over the previous mask
  frac_fixed dv_fixed / vbar_prev    (fractional rate, fixed mask; = (integral of dv/dt dA / A) / vbar)
  frac       dv_comove / vbar_prev   (fractional rate, co-moving mask; per time unit)

Usage:
    python run_v_rate_dist.py --n-frames 500
"""
import argparse, time
import numpy as np
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



def label_means(labels, n, field):
    lab = labels.ravel()
    area = np.bincount(lab, minlength=n + 1).astype(float)
    s = np.bincount(lab, weights=field.ravel(), minlength=n + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return s / area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-frames", type=int, default=500)
    ap.add_argument("--out", default="v_rate_dist")
    args = ap.parse_args()

    sim = PseudoSpectralGS.from_checkpoint(CHECKPOINT_FILE)
    proxy = GrayScottSim(n=sim.n, L=sim.L, Du=sim.Du, Dv=sim.Dv, F=sim.F, k=sim.k, dt=sim.dt)
    proxy.u, proxy.v = sim.u, sim.v
    prev_labels, prev_n = segment(proxy)
    prev_v = sim.v.copy()
    dt_frame = SAMPLE_EVERY_STEPS * sim.dt

    fixed, comove, vbar, area = [], [], [], []
    t0 = time.time()
    for fr in range(1, args.n_frames + 1):
        sim.run(SAMPLE_EVERY_STEPS)
        proxy.u, proxy.v = sim.u, sim.v
        labels, n = segment(proxy)
        if n == 0:
            break
        v = sim.v
        fwd, bwd = match_organisms_by_overlap(prev_labels, labels, prev_n, n)
        m_prev = label_means(prev_labels, prev_n, prev_v)
        m_fixed = label_means(prev_labels, prev_n, v - prev_v)
        m_curr = label_means(labels, n, v)
        a_prev = np.bincount(prev_labels.ravel(), minlength=prev_n + 1)
        for p, (c, frac) in fwd.items():
            if c is None or frac <= 0.5 or len(bwd.get(c, [])) != 1:
                continue
            fixed.append(m_fixed[p] / dt_frame)
            comove.append((m_curr[c] - m_prev[p]) / dt_frame)
            vbar.append(m_prev[p])
            area.append(a_prev[p])
        prev_labels, prev_n, prev_v = labels, n, v.copy()
        if fr % 20 == 0:
            print(f"frame {fr}/{args.n_frames} organisms={n} transitions={len(fixed)} "
                  f"{(time.time()-t0)/fr:.2f}s/frame", flush=True)

    fixed, comove, vbar, area = map(np.array, (fixed, comove, vbar, area))
    frac = comove / vbar
    frac_fixed = fixed / vbar
    np.savez(args.out + ".npz", dv_fixed=fixed, dv_comove=comove, vbar_prev=vbar,
             area_prev=area, frac=frac, frac_fixed=frac_fixed)
    print(f"\nN={len(fixed)} transitions, dt/frame={dt_frame}")
    for name, x in (("dv_fixed", fixed), ("dv_comove", comove), ("|dv_comove|", np.abs(comove)),
                    ("vbar", vbar), ("dv_fixed/vbar", frac_fixed), ("dv_comove/vbar", frac), ("|dv_comove/vbar|", np.abs(frac))):
        q = np.percentile(x, [1, 10, 50, 90, 99])
        print(f"  {name:18s} mean={x.mean():.4g} sd={x.std():.4g} pct[1,10,50,90,99]={np.round(q,5)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for a, x, lab, col in ((ax[0], frac_fixed, r"$\left(\int \partial_t v\, dA / A\right)/\langle v\rangle$ (fixed mask)", "#4477aa"),
                           (ax[1], frac, r"$\dot{\langle v\rangle}/\langle v\rangle$ (co-moving mask)", "#cc6677")):
        lim = np.percentile(np.abs(x), 99.5)
        a.hist(x, bins=np.linspace(-lim, lim, 81), color=col)
        a.axvline(0, color="k", ls=":", lw=1); a.axvline(np.median(x), color="k", ls="--", lw=1)
        a.set_xlabel(lab)
    ax[0].set_ylabel("organism-transitions")
    fig.tight_layout(); fig.savefig(args.out + ".png", dpi=200)
    print("saved", args.out)


if __name__ == "__main__":
    main()
